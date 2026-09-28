from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests import test_reward_shaping as shaping_tests
from verifier_rl import evaluation_recovery as recovery
from verifier_rl.reward_shaping import arm_plan, execution_receipt, reserve_batch, training_evidence


def new_ids(report, prefix):
    result = deepcopy(report)
    for suite in result['suites']:
        for outcome in suite['outcomes']:
            for attempt in outcome['attempts']:
                if 'sandbox_id' in attempt['metadata']:
                    attempt['metadata']['sandbox_id'] = prefix + attempt['metadata']['sandbox_id']
    return result


def infra_report(report, sid, cleanup='unconfirmed:TimeoutError'):
    result = deepcopy(report)
    suite = result['suites'][1]
    outcome = suite['outcomes'][0]
    was_passed = outcome['passed'] is True
    outcome.update(passed=None, reason='infrastructure_error', actual=None)
    attempt = outcome['attempts'][0]
    attempt.update(status='infrastructure_error', detail='cleanup_unconfirmed')
    metadata = attempt['metadata']
    for key in ('preflight_returncode', 'returncode', 'stdout_bytes', 'stdout_preview', 'stdout_sha256'):
        metadata.pop(key, None)
    metadata.update(sandbox_id=sid, preflight_stage='command_start', cleanup=cleanup)
    suite.update(all_passed=None, reward=None, infrastructure_errors=1,
                 passed_count=suite['passed_count'] - int(was_passed))
    return result


class EvaluationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shaping_tests.RewardShapingTests.setUpClass()
        prior = shaping_tests.RewardShapingTests
        generations = deepcopy(prior.generations)
        for formula, generation in generations.items():
            generation['run_id'] = recovery.SOURCE_RUN + '-' + formula.replace('_', '-')
            generation['plan'] = arm_plan(prior.plan, formula, recovery.SOURCE_RUN)
            generation['training_evidence'] = training_evidence(generation['metrics'])
        cached = {f'baseline-{seed}': deepcopy(prior.reports['baseline'][i])
                  for i, seed in enumerate(range(10000, 10005))}
        failed = infra_report(prior.reports['baseline'][5], recovery.QUARANTINED_SANDBOX)
        spent = {}
        for formula, generation in generations.items():
            for group in generation['rollout_records']:
                for sample, report in zip(group['samples'], group['reports']):
                    key = f"{recovery.SOURCE_RUN}-{formula.replace('_', '-')}-training-{sample['seed']}"
                    spent[key] = execution_receipt(key, report)
        for seed in range(10000, 10005):
            key = f'{recovery.SOURCE_RUN}-baseline-before-{seed}'
            spent[key] = execution_receipt(key, cached[f'baseline-{seed}'])
        key = f'{recovery.SOURCE_RUN}-baseline-before-10005'
        cls.inputs = {'plan': deepcopy(prior.plan), 'generations': generations,
            'setup': {'sandbox_image_id': 'im-test', 'app_name': 'fixture'}, 'conformance': {},
            'cached': cached, 'failed_report': failed, 'old_budget': deepcopy(prior.budget),
            'failed_intent': {'sample': deepcopy(generations['partial']['samples'][5]),
                             'spending_reservation': reserve_batch(prior.budget, spent, key, 432)}}
        cls.budget = recovery.recovery_budget(cls.inputs, {'metered_cost': '1.92557826'}, shaping_tests.RATES)
        cls.reports = {}
        for key, policy, sample in recovery.entries(generations):
            report = prior.reports[policy][sample['seed'] - 10000]
            cls.reports[key] = deepcopy(cached[key]) if key in cached else new_ids(report, 'recovered-')

    def accounting(self):
        receipts, reservations, attempts = {}, {}, {}
        for key, _, _ in recovery.entries(self.inputs['generations']):
            if key in self.inputs['cached']:
                continue
            attempt_key = key + '-attempt-1'
            report = self.reports[key]
            receipt = execution_receipt(attempt_key, report)
            reservations[attempt_key] = reserve_batch(self.budget, receipts, attempt_key, receipt['executions'])
            receipts[attempt_key], attempts[attempt_key] = receipt, report
        return receipts, reservations, attempts

    def test_fixed_population_and_no_model_changes(self):
        self.assertEqual(len(recovery.validate_inputs(self.inputs)), 24)
        changed = deepcopy(self.inputs)
        changed['generations']['partial']['plan']['learning_rate'] = .1
        with self.assertRaises(ValueError): recovery.validate_inputs(changed)
        changed = deepcopy(self.inputs)
        changed['failed_intent']['spending_reservation']['total_reserved_usd'] = '0'
        with self.assertRaises(ValueError): recovery.validate_inputs(changed)

    def test_twenty_dollars_is_total_and_failed_reservation_remains(self):
        self.assertEqual(self.budget['total_trial_limit_usd'], '20')
        self.assertGreater(float(self.budget['fixed_reserved_usd']),
                           float(self.inputs['failed_intent']['spending_reservation']['total_reserved_usd']))
        self.assertEqual(self.budget['max_sandbox_executions'], 21 * 432)
        with self.assertRaises(ValueError):
            recovery.recovery_budget(self.inputs, {'metered_cost': '19'}, shaping_tests.RATES)

    def test_retry_requires_infra_and_confirmed_cleanup_not_candidate_signal(self):
        sample = self.inputs['failed_intent']['sample']
        bad = deepcopy(self.inputs['failed_report'])
        self.assertFalse(recovery.retry_eligible(sample, bad, 'im-test'))
        terminal = {recovery.QUARANTINED_SANDBOX: 137}
        self.assertTrue(recovery.retry_eligible(sample, bad, 'im-test', terminal))
        self.assertFalse(recovery.retry_eligible(sample, self.reports['baseline-10005'], 'im-test'))
        bad['suites'][0]['outcomes'][0]['attempts'][0]['metadata']['returncode'] = 137
        self.assertFalse(recovery.retry_eligible(sample, bad, 'im-test', terminal))

    def test_failed_batch_retains_full_reservation(self):
        receipt = recovery.failed_receipt('failed', self.inputs['failed_report'])
        self.assertEqual(receipt['accounted_sandbox_seconds'], 51840)
        self.assertTrue(receipt['executions_are_reserved_upper_bound'])

    def test_full_offline_comparison_and_spending_recompute(self):
        receipts, reservations, attempts = self.accounting()
        result = recovery.verify_recovery(self.inputs, self.reports, self.budget,
                                           receipts, reservations, attempts, {})
        self.assertEqual(result['recovery']['new_model_samples'], 0)
        self.assertEqual(result['recovery']['new_program_evaluations'], 19)
        self.assertEqual(result['recovery']['all_recorded_sandbox_executions_including_failed_batch'], 12208)
        self.assertEqual(result['policies']['baseline']['full_audit_passes'], 8)
        changed = deepcopy(receipts)
        changed[next(iter(changed))]['report_hash'] = 'tampered'
        with self.assertRaises(ValueError):
            recovery.verify_recovery(self.inputs, self.reports, self.budget, changed, reservations, attempts, {})

    @unittest.skipUnless(find_spec('modal') is not None, 'optional Modal SDK needed for mocked launchers')
    def test_warm_controller_uses_cached_results_and_retries_only_infra(self):
        import modal_grpo_pilot as launcher
        from verifier_rl.cli import create_run_directory as create_local_directory
        rows = [row for row in recovery.entries(self.inputs['generations']) if row[0] not in self.inputs['cached']]
        first = rows[0][0]
        failed = infra_report(new_ids(self.reports[first], 'failed-new-'), 'sb-confirmed-ended', 'terminated')
        responses = [failed] + [self.reports[key] for key, _, _ in rows]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with (patch.object(launcher, 'Path', side_effect=lambda p: base / str(p).removeprefix('/artifacts/')),
                  patch.object(launcher, 'create_run_directory', side_effect=lambda p:
                               create_local_directory(str(base / str(p).removeprefix('/artifacts/')))),
                  patch.object(launcher, 'artifacts'), patch.object(launcher, 'require_current_conformance'),
                  patch.object(launcher, 'ModalBackend'),
                  patch('builtins.print'),
                  patch.object(launcher, 'train_and_generate') as gpu,
                  patch.object(launcher, 'evaluate_submission', new=AsyncMock(side_effect=deepcopy(responses))) as evaluate):
                result = launcher.recover_evaluation.local('qwen-recovery-unit-test', self.inputs,
                                                            self.budget, time.time() + 3600, {})
                gpu.remote.assert_not_called()
                gpu.spawn.assert_not_called()
                self.assertEqual(evaluate.await_count, 20)
            self.assertEqual(result['reports']['baseline-10000'], self.inputs['cached']['baseline-10000'])
            self.assertEqual(result['summary']['recovery']['automatic_retries'], 1)
            self.assertTrue((base/'qwen-recovery-unit-test'/f'{first}-attempt-1'/'result.json').exists())
            self.assertEqual(len(result['reports']), 24)


if __name__ == '__main__':
    unittest.main()
