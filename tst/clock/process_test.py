"""Who is faked, the marker that keeps a program real, the environment read once at load, and fork."""


def test_only_the_session_users_processes_follow_an_armed_page(gate):
    assert gate("identity") == {"unarmed_session": "real", "session": "universe", "root": "real", "other_user": "real"}


def test_a_marked_program_stays_real_and_loses_the_marker_so_its_children_are_faked(gate):
    r = gate("marker")
    assert r["rc"] == 0, r
    assert r["lines"] == {"unmarked_date": "universe", "marked_bash_itself": "real", "marked_bash_child_date": "universe",
                          "marked_bash_grandchild_sh_date": "universe", "marked_bash_exec_child": "universe",
                          "marked_bash_environ": 0.0, "child_environ": 0.0, "marked_python_itself": "real",
                          "marked_python_child": "universe"}


def test_setenv_racing_clock_reads_and_timed_waits_never_crashes_it(gate):
    assert gate("envrace", 10.0) == [{"rc": 0, "out": "survived"}] * 3


def test_a_fork_while_another_thread_closes_a_timerfd_never_leaves_the_child_stuck(gate):
    """1,000 forks while another thread duplicates and closes a timerfd, each child closing its descriptors before it
    exits: the library's lock, which that thread may hold at the fork, is free in every child."""
    assert gate("forklock", 1000) == {"forks": 1000, "stuck": 0}
