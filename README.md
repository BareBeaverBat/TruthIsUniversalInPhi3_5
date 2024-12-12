# TruthIsUniversalInPhi3_5
Reproducing several of the experiments of the ["Truth is Universal" paper (Bürger et al.)](https://arxiv.org/abs/2407.12831v2) for [Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), and testing out 
a few things that they didn't mention trying.

[Final report](Report.md)

## Analysis Plan

### Different layers
Out of Phi-3.5-mini's 32 'layers' (i.e. transformer decoder blocks), I retrieved the activations (the residual stream) 
for the last token in a sequence after the embedding layer (i.e. before the first decoder block) and after every decoder block.

Following Bürger et al., I measured the 'separation' of the true and false statements for each layer's hidden states 
(definition in caption of Figure 2 of that paper, reproduced below^).  
I computed this average separation for a given layer over all datasets (rather than just 4).   

I found a multi-peak pattern when looking across different datasets, even though there was one clear global peak when averaging across all datasets. 
I trained truth-and-polarity directions on the overall peak layer, one additional later layer that was near several dataset-specific peaks, and on the concatenation of those two layers' activations.  
The rationale for the combination of the global peak layer and a local peak layer is that representations tend to vary in abstractness as one passes through the layers, and I had an intuition that it might be useful to combine both very-high-abstraction representations and medium-abstraction representations when linear-probing for polarity and truth directions.

^Ratio of the between-class variance and within-class variance of activations corresponding to true and false statements, across residual stream layers, averaged over all dimensions of the respective layer

### Datasets usage
I used the [datasets](https://github.com/sciai-lab/Truth_is_Universal/tree/main/datasets) from the Truth is Universal paper.

1 caveat- in the "facts" and "neg_facts" datasets, there were a total of 6 statements which put a single or double
quote character after the period at the end of the statement. This is specific to American-English grammar and is very
inconvenient for the data analysis (can't rely on the end of each statement being an end punctuation character).  
As a result, I swapped the order of the last 2 characters in 4 of those statements (the statements at 0-based indexes 51 and 85 in both datasets). 
While looking at the last 2 of those statements (at 0-based index 482 in both datasets), I concluded that there was a typo:  
`The planet Mars [is/isn't] known as the Red Planet" due to its reddish appearance."`  
It doesn't make any sense to have a double quote around the phrase " due to its reddish appearance" include the space character before the word `due`.
Therefore, I moved the terminal double quote to be the beginning of a quoted title `"Red Planet"`  
`The planet Mars [is/isn't] known as the "Red Planet" due to its reddish appearance.`

#### Topics with 4 dataset variants each

The following topics have all 4 variants (affirmative, negated, conjunction, disjunction) that're used for training, plus 2 later-added variants that are only used for testing (german affirmative and german negated):
- animal class
- cities
- element symbols
- facts
- inventors
- spanish-english translation

Following the paper, I trained 
- a set of directions for each topic's affirmative statements
- a set of directions for each topic's affirmative and negated statements
- a set of directions on all 4 variants of a topic's statements

and I also trained
- a set of directions for each topic's conjunctive statements
- a set of directions for each topic's disjunctive statements
- a set of directions for each topic's affirmative and negated statements
- a set of directions for each topic's affirmative and disjunctive statements
- a set of directions for each topic's negated and conjunctive statements

When learning truth directions from multiple topics at once, I trained a set of directions on the topics 
"animal class", "facts", and "inventors";  
The topics "cities", "element symbols", and "spanish-english translation" were used as test sets in that part of the analysis.  
In that multi-topic scenario, I included the affirmative, negated, and conjunctive variants from each such topic and left the disjunctive variants in each such topic as additional test sets.

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

For 'real world scenarios', I trained a set of directions on "unambiguous lie", "unambiguous truthful reply", and 
"ambiguous truthful reply", leaving "ambiguous lie" and "honest reply despite incentive to lie" as a test set.

Finally, I trained a set of truth directions on all of 
- affirmative and negated statements from "animal class", "element symbols", "facts", and "inventors"
- "unambiguous lie", "unambiguous truthful reply", and "ambiguous truthful reply" from 'real world scenarios'
- "common claim" from 'true false'
- "smaller than" from 'relative comparison'

I also used the non-4-variants topics as additional test sets for the various learned truth directions where it made sense.

### Training of truth and polarity directions
Whenever training T&P directions for a given choice of layer(s) and dataset(s), I of course did an 80-20% split 
of the activations for that choice and trained on the 80% of activations, using the performance of the learned directions 
on the held-out 20% to confirm that the training process went as intended. 

