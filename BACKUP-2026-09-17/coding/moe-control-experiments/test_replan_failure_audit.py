import unittest

from audit_replan_failures import completed_failure_parents, summarize_pairs


class FailureScopeTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(parents=[dict(name='failed', source_success=False),
                                    dict(name='successful', source_success=True)],
                           replicates=2, arms=['native10', 'window5'])
        self.rows = [dict(parent='failed', replicate=r, arm=arm, source_success=False)
                     for r in range(2) for arm in self.config['arms']]

    def test_complete_failed_subset(self):
        self.assertEqual([p['name'] for p in completed_failure_parents(self.config, self.rows)], ['failed'])

    def test_cannot_drop_a_failed_branch(self):
        with self.assertRaises(AssertionError):
            completed_failure_parents(self.config, self.rows[:-1])

    def test_cannot_duplicate_or_include_success(self):
        for extra in (self.rows[0], dict(parent='successful', replicate=0, arm='native10', source_success=True)):
            with self.assertRaises(AssertionError):
                completed_failure_parents(self.config, self.rows + [extra])

    def test_count_rescue_against_paired_baseline(self):
        pairs = [dict(native_success=False, window_success=True, rescue=True, harm=False, gain=1),
                 dict(native_success=True, window_success=False, rescue=False, harm=True, gain=-1),
                 dict(native_success=False, window_success=False, rescue=False, harm=False, gain=0)]
        result = summarize_pairs(pairs)
        self.assertEqual(result['pairs'], 3)
        self.assertEqual(result['rescues'], 1)
        self.assertEqual(result['native_success_window_failure'], 1)
        self.assertEqual(result['paired_success_gain'], 0)


if __name__ == '__main__':
    unittest.main()
