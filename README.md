# TruthIsUniversalInPhi3_5
Reproducing most of the findings of the "Truth is Universal" paper for Phi-3.5-mini, and testing out a few things that 
they didn't mention trying.

## Analysis Plan

### Different layers
Out of Phi-3.5-mini's 32 'layers' (i.e. transformer decoder blocks), I'll sample the activations (the residual stream) 
after the 6th (~20%), 16th (50%), 22nd (~70%), and 29th (~90%) layers.

I'll explore both the training of truth-and-polarity-directions on one layer at a time and the training of such 
directions based on vectors that are the concatenation of two layers' activations.

As a result, whenever a line in the following section speaks of 'a set of directions', it means 10 sets of directions 
for groupings of a) activation vectors from different layers or b) concatenations of activation vectors from a pair of layers
- 1 set for the post-6-layer activations
- 1 set for the post-16-layer activations
- 1 set for the post-22-layer activations
- 1 set for the post-29-layer activations
- 1 set for the post-6 + post-16 activations
- 1 set for the post-6 + post-22 activations
- 1 set for the post-6 + post-29 activations
- 1 set for the post-16 + post-22 activations
- 1 set for the post-16 + post-29 activations
- 1 set for the post-22 + post-29 activations.

### Datasets usage
I'll be using the [datasets](https://github.com/sciai-lab/Truth_is_Universal/tree/main/datasets) from the 
Truth is Universal paper.

#### Topics with 4 dataset variants each

The following topics have all 4 variants (affirmative, negated, conjunction, disjunction):
- animal class
- cities
- element symbols
- facts
- inventors
- spanish-english translation

Following the paper, I'll train 
- a set of directions for each topic's affirmative statements
- a set of directions for each topics affirmative and negated statements
- a set of directions on all 4 variants of a topic's statements

and I'll also train
- a set of directions for each topic's affirmative and disjunctive statements
- a set of directions for each topic's negated and conjunctive statements

When learning truth directions from multiple topics at once, I'll train a set of directions on the topics 
"animal class", "element symbols", "facts", and "inventors"; 
The topics "cities" and "spanish-english translation" will be used as test sets in that part of the analysis.

#### Other datasets

There are also these datasets which don't follow the 4-variants pattern.
- real world scenarios: 
  - unambiguous lie 
  - unambiguous truthful reply
  - honest reply despite incentive to lie
  - ambiguous lie
  - ambiguous truthful reply
- true false: 
  - common claim
  - counterfactual
- relative comparison: 
  - larger than
  - smaller than

For 'real world scenarios', I'll train a set of directions on "unambiguous lie", "unambiguous truthful reply", and 
"ambiguous truthful reply", leaving "ambiguous lie" and "honest reply despite incentive to lie" as a test set.

For 'true false', I'll train a set of directions on "common claim" and test them on "counterfactual".

For relative comparison, I'll train a set of directions on "smaller than" and test them on "larger than".

Finally, I'll train a set of truth directions on all of 
- affirmative and negated statements from "animal class", "element symbols", "facts", and "inventors"
- "unambiguous lie", "unambiguous truthful reply", and "ambiguous truthful reply" from 'real world scenarios'
- "common claim" from 'true false'
- "smaller than" from 'relative comparison'

I'll also use the non-4-variants topics as additional test sets for the various learned truth directions where it makes sense.

### Training of truth and polarity directions
Whenever training T&P directions for a given choice of layer(s) and dataset(s), I will of course do an 80-20% split 
of the activations for that choice and train on the 80% of activations, using the performance of the learned directions 
on the held-out 20% to confirm that the training process went as intended. 

