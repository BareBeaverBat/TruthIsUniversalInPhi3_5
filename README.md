# TruthIsUniversalInPhi3_5
Reproducing several of the experiments of the 
["Truth is Universal" paper (Bürger et al.)](https://arxiv.org/abs/2407.12831v2) for 
[Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), and testing out a few things that they didn't 
mention trying.

[Final report](Report.md)

As described in the report, some details in this document have been edited in version 2. Cf.
[the original README](https://github.com/BareBeaverBat/TruthIsUniversalInPhi3_5/blob/v1/README.md)

## Analysis Plan

### Different layers
Out of Phi-3.5-mini's 32 'layers' (i.e. transformer decoder blocks), I retrieved the activations (the residual stream) 
for the last token in a sequence after the embedding layer (i.e. before the first decoder block) and after every decoder 
block.

Following Bürger et al., I measured the 'separation' of the true and false statements for each layer's hidden states 
(definition in caption of Figure 2 of that paper, reproduced below^).  
I computed this average separation for a given layer over all datasets (rather than just 4).   

I found a multi-peak pattern when looking across different datasets, even though there was one clear global peak when 
averaging across all datasets (after layer 18).  
I trained truth-and-polarity directions on the overall peak layer (18).  

^Ratio of the between-class variance and within-class variance of activations corresponding to true and false 
statements, across residual stream layers, averaged over all dimensions of the respective layer

### Datasets usage
I used the [datasets](https://github.com/sciai-lab/Truth_is_Universal/tree/main/datasets) from the Truth is Universal 
paper.

1 caveat- in the "facts" and "neg_facts" datasets, there were a total of 6 statements which put a single or double
quote character after the punctuation mark at the end of the statement. This is specific to American-English grammar and 
is very inconvenient for the data analysis (can't rely on the end of each statement being an end punctuation character).  
As a result, I swapped the order of the last 2 characters in 4 of those statements (the statements at 0-based indexes 51 
and 85 in both datasets).  
While looking at the last 2 of those statements (at 0-based index 482 in both datasets), I concluded that there was a 
typo:  
`The planet Mars [is/isn't] known as the Red Planet" due to its reddish appearance."`  
It doesn't make any sense to have double quotes around the phrase " due to its reddish appearance" that include the 
space character before the word `due`. Also, it makes sense to put quotes around a title like `Red Planet` but doesn't 
make sense to put them around an explanatory phrase like `due to its reddish appearance` unless that phrase was a direct 
quote from someone (which doesn't seem to be the case in this context).  
Therefore, I moved the terminal double quote to be the beginning of the quoted title `"Red Planet"`    
`The planet Mars [is/isn't] known as the "Red Planet" due to its reddish appearance.`

#### Topics with 6 dataset variants each

The following topics have all 4 main variants (affirmative, negated, conjunction, disjunction) that're used for 
training, plus 2 later-added variants that are only used for testing (german affirmative and german negated):  
- animal class
- cities
- element symbols
- facts
- inventors
- spanish-english translation

Following the paper, I trained 
- a set of directions for each topic's affirmative and negated statements
- a set of directions on all 4 variants of a topic's statements

and I also trained
- a set of directions for each topic's negated and conjunctive statements
- a set of directions for each topic's negated and disjunctive statements

When learning truth directions from multiple topics at once, I trained a set of directions on the topics 
"animal class", "facts", and "inventors";  
In two multi-topic scenarios, I included the affirmative and negated variants from each such topic while in a third
I also included the conjunctive variant.

The topics "cities", "element symbols", and "spanish-english translation" were used as test sets in that part of the 
analysis.  

#### Other dataset categories

There are also these dataset categories which don't follow the 6-variants pattern.
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

The 'real world scenarios' datasets were all very small and contained statements that were all of the same polarity, so 
they were left as test sets.

One of the multi-topic scenarios had the "smaller than" relative comparisons data and the "common claim" true-false data 
added to the affirmative and negated variants of the 3 training topics.

I also used the non-6-variants categories as additional test sets for the various learned truth directions where it made 
sense.

### Training of truth and polarity directions
Whenever training T&P directions and probes for a given choice of layer(s) and dataset(s), I did an 80-20% split 
of the activations for that choice and trained on the 80% of activations, using the performance of the learned 
directions on the held-out 20% to confirm that the training process went as intended. 

## Reproduction
A Python 3.12 interpreter was used for local Jupyter notebook execution.

1. The [create_data_index.ipynb](create_data_index.ipynb) script can be run locally to produce an index of datasets 
(which must then be committed and pushed before step 2).
2. Activation harvesting can then be done by running the
[harvesting_activations_of_phi_3_5_mini.ipynb](harvesting_activations_of_phi_3_5_mini.ipynb) Jupyter notebook on Google 
Colab with an Nvidia A100 GPU after uploading a zip of the 
[Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) HuggingFace files to one's Google Drive.
3. The [compute_t_f_sep_by_layer.ipynb](compute_t_f_sep_by_layer.ipynb) Jupyter notebook can be run locally to explore
the separation of true and false statements in the activations of different layers for these datasets and this model, 
then to separate out just the layer 18 activations in one file folder.
4. The [split_train_validation.ipynb](split_train_validation.ipynb) Jupyter notebook can be run locally to create
the train-validation splits for all datasets.
5. The [training_probes.ipynb](training_probes.ipynb) Jupyter notebook can be run locally to train TTPD probes (
including the learning of truth and polarity directions) and baseline linear probes for each scenario and evaluate them 
on the train and validation splits of their scenario's data.
6. The [evaluating_generalization_of_truth_directions.ipynb](evaluating_generalization_of_truth_directions.ipynb)
Jupyter notebook can be run locally to record evaluations of the probes on 'test' datasets which hadn't been included 
in their training.
7. The [analyzing_overall_results.ipynb](analyzing_overall_results.ipynb) Jupyter notebook can be run locally to compute 
various aggregating analyses of the resulting statistics.
