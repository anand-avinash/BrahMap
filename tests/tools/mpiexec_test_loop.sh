#!/bin/bash

# Color formats
bbred='\033[1;91m'   # bold bright red
bbgreen='\033[1;92m' # bold bright green
nc='\033[0m'         # no color

# To print formatted text in a block
formatted_print() {
  local print_string="$1"
  local nprocs="$2"

  printf '\n\n\n\n%s \n%s\n%s \n\n\n\n' \
  "$(printf '=%.0s' {1..36})" "$print_string" "$(printf '=%.0s' {1..36})"
}

# String to collect the failing nprocs
error_nprocs=()

# Testing the execution for different nprocs
for nprocs in 1 2 5 6; do

  formatted_print "Running test with nprocs = $nprocs" "$nprocs"
  
  if ! mpiexec -n $nprocs pytest; then
    # if fails, prints the status and stores the `nprocs`` in `error_nprocs`
    formatted_print \
    "Test status for nprocs = $nprocs: $(printf "${bbred}FAILED${nc}")"\
    "$nprocs"
    
    error_nprocs+=("$nprocs")
  else
    # if passed, prints the status
    formatted_print \
    "Test status for nprocs = $nprocs: $(printf "${bbgreen}PASSED${nc}")"\
    "$nprocs"
  fi

done

if [ ${#error_nprocs[@]} -ne 0 ]; then
  # exit 0, when some tests fail
  formatted_print \
  "$(printf "${bbred}Test failed for nproc(s): ${error_nprocs[*]}${nc}")"\
  "$error_nprocs"

  exit 1
else
  # when all tests are passing
  formatted_print "$(printf "${bbgreen}Test passed for all nprocs${nc}")"

  exit 0
fi
