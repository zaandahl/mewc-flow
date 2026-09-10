# Frozen previous runtime; this release replaces its source with reviewed fixes.
FROM zaandahl/mewc-flow:2.0.3@sha256:602d7738b0cd1141a2292fcf549c4c69eac808c64ed5efccac3ebf04a97306f6
WORKDIR /code
COPY src/ .
