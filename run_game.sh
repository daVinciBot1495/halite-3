#!/bin/sh

set -e

kotlinc MyBot.kt hlt/*.kt -include-runtime -d MyBot.jar
./halite --replay-directory replays/ -vvv --width 32 --height 32 --no-replay --no-logs "kotlin -classpath MyBot.jar MyBot train" "kotlin -classpath MyBot.jar MyBot"
