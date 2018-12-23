#!/bin/sh

set -e

kotlinc MyBot.kt hlt/*.kt -include-runtime -d MyBot.jar
./halite --replay-directory replays/ -vvv --width 32 --height 32 "kotlin -classpath MyBot.jar MyBot" "kotlin -classpath MyBot.jar MyBot"
