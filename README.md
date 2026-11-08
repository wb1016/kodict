# DO NOT USE
## Why mining instead of asking an LLM

The first version of this project generated words and bigrams with an LLM.
It produced *invented* word pairs with no notion of spacing or grammar.
the highest-scoring suggestion after `사람` was `이`,
which the keyboard inserted with a space as `사람 이` (invalid Korean: particles attach without a space),
and the next-word chain then looped `사람 이 사람 이 사람 이 ...` forever.
