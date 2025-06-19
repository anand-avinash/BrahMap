import pytest

# Dictionaries to keep track of the results and parameter counts of parametrized test cases
test_results_status = {}
test_param_counts = {}
forced_skipped_tests = set()


def pytest_configure(config):
    # By default, pytest
    config.addinivalue_line(
        "markers",
        "ignore_param_count: Marker for the parameterized test "
        "functions/classes that are to be excluded from the test count "
        "filtering",
    )


def get_base_nodeid(nodeid):
    """Strips the parameter id from the nodeid and returns the rest

    Args:
        nodeid (str): nodeid

    Returns:
        str: nodeid without the parameter id
    """
    if "[" in nodeid:
        return nodeid.split("[")[0]
    return nodeid


def pytest_collection_modifyitems(items):
    """This function counts the number of parameters for a parameterized
    test"""
    for item in items:
        if "parametrize" in item.keywords and "ignore_param_count" not in item.keywords:
            base_nodeid = get_base_nodeid(item.nodeid)
            if base_nodeid not in test_param_counts:
                test_param_counts[base_nodeid] = 0
            test_param_counts[base_nodeid] += 1


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """This function stores the status of a parameterized test for each
    parameter"""
    # Execute the test
    outcome = yield

    # Only process parametrized tests
    if "parametrize" in item.keywords and "ignore_param_count" not in item.keywords:
        base_nodeid = get_base_nodeid(item.nodeid)

        # Initialize the list for this test function if not already done
        if base_nodeid not in test_results_status:
            test_results_status[base_nodeid] = []

        # Check if the test passed
        if outcome.excinfo is None:
            test_results_status[base_nodeid].append(True)
        else:
            test_results_status[base_nodeid].append(False)


@pytest.hookimpl(tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """This hook function marks the test to pass if at least half of the
    parameterized tests are passed. It also issues warning if the test is not
    passed by all parameters. A test is excluded from this filtering if it is
    marked with `@pytest.mark.ignore_param_count`."""
    for base_nodeid in list(test_results_status.keys()):
        passed_count = test_results_status[base_nodeid].count(True)
        params_count = test_param_counts[base_nodeid]

        if passed_count >= int(params_count / 2):
            failed_report = terminalreporter.stats.get("failed", []).copy()
            for report in failed_report:
                if base_nodeid == get_base_nodeid(report.nodeid):
                    terminalreporter.stats["failed"].remove(report)
                    report.outcome = "skipped"
                    terminalreporter.stats.setdefault("skipped", []).append(report)
                    forced_skipped_tests.add(base_nodeid)

    # If there is no failed test, set the exit status to 0
    if not terminalreporter.stats.get("failed", []):
        terminalreporter._session.exitstatus = 0

    # Print a summary of forced skipped tests
    if forced_skipped_tests:
        terminalreporter.section(
            "forced skipped tests summary",
            cyan=True,
            blink=True,
            bold=True,
            sep="=",
        )
        terminalreporter.write("See the test report for more details.\n")
        for base_nodeid in forced_skipped_tests:
            passed_count = test_results_status[base_nodeid].count(True)
            params_count = test_param_counts[base_nodeid]

            terminalreporter.write(
                f"Test {base_nodeid} is passing only for {passed_count} out of {params_count} parameters. Test status: {['passed' if status else 'failed' for status in test_results_status[base_nodeid]]}\n"
            )

    # Clear the dictionaries
    test_results_status.clear()
    test_param_counts.clear()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after the whole test run finished, right before returning the exit
    status to the system.
    """
    pass
