from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score


class TestScoreParity:
    def test_all_pass_aggregation(self):
        flat = [{"score": 1.0, "result": True}, {"score": 1.0, "result": True}]
        assert aggregate_score(flat, ScoreAggregator.ALL_PASS) == 1.0

    def test_all_pass_partial_fail(self):
        flat = [{"score": 1.0, "result": True}, {"score": 0.0, "result": False}]
        assert aggregate_score(flat, ScoreAggregator.ALL_PASS) == 0.0

    def test_any_pass_one_pass(self):
        flat = [{"score": 0.0, "result": False}, {"score": 1.0, "result": True}]
        assert aggregate_score(flat, ScoreAggregator.ANY_PASS) == 1.0

    def test_any_pass_all_fail(self):
        flat = [{"score": 0.0, "result": False}, {"score": 0.0, "result": False}]
        assert aggregate_score(flat, ScoreAggregator.ANY_PASS) == 0.0

    def test_empty_results(self):
        assert aggregate_score([], ScoreAggregator.ALL_PASS) == 0.0
