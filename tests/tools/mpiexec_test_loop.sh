#!/bin/bash

# Primary purpose of this script is to run the test using pytest on the BrahMap
# package. It runs the test for multiple number of MPI processes in a bash
# loop. In a simple case, the loop will terminate with an error if the test
# fails for an iteration. In this test instead, the loop will simply continue
# despite catching an error while recording the passing status of each
# iteration. Once the loop is over, the script will throw an error if MORE
# THAN TWO loop iterations fails. Otherwise, the script will terminate
# normally.

# Color formats
bbred='\033[1;91m'   # bold bright red
bbgreen='\033[1;92m' # bold bright green
nc='\033[0m'         # no color

# To print formatted text in a block
formatted_print() {
  local print_string="$1"

  printf '\n\n\n\n%s \n%s\n%s \n\n\n\n' \
    "$(printf '=%.0s' {1..40})" "$print_string" "$(printf '=%.0s' {1..40})"
}

# String to collect the failing nprocs
error_nprocs=()

# Testing the execution for different nprocs
for nprocs in 1 2 5 6; do

  formatted_print "Running test with nprocs = $nprocs"

  if ! mpiexec --map-by :OVERSUBSCRIBE -n $nprocs pytest; then
    # if fails, prints the status and stores the `nprocs`` in `error_nprocs`
    formatted_print \
      "Test status for nprocs = $nprocs: $(printf "${bbred}FAILED${nc}")"

    error_nprocs+=("$nprocs")
  else
    # if passed, prints the status
    formatted_print \
      "Test status for nprocs = $nprocs: $(printf "${bbgreen}PASSED${nc}")"
  fi

done

num_errors=${#error_nprocs[@]}

if [ ${num_errors} -gt 2 ]; then
  # exit 1, when more than two tests fail
  formatted_print \
    "$(printf "${bbred}Test failed for nproc(s): ${error_nprocs[*]}${nc}")"
  formatted_print "$(printf "${bbred}Overall status: FAILED${nc}")"

  exit 1
else
  # exit 0, when two or more tests are passing
  if [ ${num_errors} -ne 0 ]; then
    formatted_print \
      "$(printf "${bbgreen}Test passed for all nprocs except: ${error_nprocs[*]}${nc}")"
  else
    formatted_print "$(printf "${bbgreen}Test passed for all nprocs${nc}")"
  fi

  formatted_print "$(printf "${bbgreen}Overall status: PASSED${nc}")"

  exit 0
fi
