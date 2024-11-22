#!/bin/sh

export JAVA_HOME="/usr/lib/jvm/java-17-openjdk/"
export PATH="/usr/lib/jvm/java-17-openjdk/bin/:$PATH"

cd "$(dirname "$0")"

source ./activate.sh 

/usr/bin/clojure "$@"

