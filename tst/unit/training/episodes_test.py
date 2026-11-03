"""The arguments a rollout's tool calls run with, typed by each tool's schema, and the run groups a trainer files its
episodes in by training step."""

from agent_env.training._episodes import RunGroup, TrainingSteps, run_group_id, typed_arguments

LOOKUP = {"type": "object", "properties": {
    "zip": {"type": "string"},
    "note": {"type": ["string", "null"]},
    "order": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "count": {"type": "integer"},
    "price": {"type": "number"},
    "gift": {"type": "boolean"},
    "items": {"type": "array", "items": {"type": "string"}},
}}

def test_a_scalar_its_schema_wants_as_a_string_is_given_back_as_the_string_it_was_written_as():
    assert typed_arguments({"zip": 19122, "note": 7, "order": True}, LOOKUP) == {
        "zip": "19122", "note": "7", "order": "true"}


def test_values_their_schema_allows_are_left_as_parsed():
    arguments = {"zip": "19122", "note": None, "count": 3, "price": 2, "gift": False, "items": ["1151293680"]}
    assert typed_arguments(arguments, LOOKUP) == arguments


def test_an_argument_the_schema_doesnt_name_and_arguments_that_arent_an_object_are_left_alone():
    assert typed_arguments({"other": 5}, LOOKUP) == {"other": 5}
    assert typed_arguments([19122], LOOKUP) == [19122]


def test_a_group_is_filed_under_its_run_phase_and_step_and_an_evaluation_at_the_step_it_measures():
    steps = TrainingSteps()
    assert steps.evaluated() == 0
    assert [steps.trained(index) for index in range(4)] == [0, 1, 2, 3]
    assert steps.evaluated() == 4
    assert run_group_id("retail-run2", "eval", steps.evaluated()) == "retail-run2/eval/step-0004"


def test_an_evaluations_group_made_once_is_filed_at_the_step_each_time_it_is_played():
    steps = TrainingSteps()
    evaluation = RunGroup("retail-run2", steps)
    assert evaluation.id() == "retail-run2/eval/step-0000"
    trained = RunGroup("retail-run2", steps, steps.trained(0))
    assert (trained.id(), evaluation.id()) == ("retail-run2/train/step-0000", "retail-run2/eval/step-0001")
