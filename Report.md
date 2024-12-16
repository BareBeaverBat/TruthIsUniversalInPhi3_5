# Summary

In this project, I reproduced some of the main results of the paper 
["Truth is Universal" paper (Bürger et al.)](https://arxiv.org/abs/2407.12831v2) for 
[Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
without looking at the source code for Bürger et al.'s paper until it was time to write this report.

I  
1. harvested residual stream activations after all layers for the last token of each statement in Bürger et al.'s 
datasets
2. analyzed truth-falsehood separation in those activations by layer across all datasets and picked 2 layers to focus on
3. simultaneously learned truth and polarity directions in the vector spaces of activations after those 2 layers and the
vector space of the concatenation of activations after those 2 layers.
4. trained TTPD probes that used those directions and evaluated their performance on unseen datasets
5. trained linear probes as a baseline (mimicking the linear probes of 
[Marks & Tegmark](https://arxiv.org/abs/2310.06824), which Bürger et al. had used as one of their baselines) and 
compared their performance to the TTPD probes


## Findings

1. Truth probes of Phi-3.5-mini's activations did markedly worse than probes of relatively larger (7B or 8B parameters) 
LLMs' activations on topics involving a lot of memorized-trivia/world-knowledge.
   1. Phi-3.5-mini only has 3.8B parameters. In fact, the [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) (in 
   section 6- Weakness) called out this problem (that the phi-3-mini model, with the same number of weights as 
   phi-3.5-mini, doesn't have as much capacity for storing 'factual knowledge' as models with 2x or more parameters)  
2. Simple linear probes did better than TTPD probes for Phi-3.5-mini (as trained by this project), by ~10pp (percentage 
points) on validation accuracy and ~4pp on test accuracy.


## Retrospective

One downside of the "reproduce their work without looking at their code" approach was that I misunderstood Bürger et 
al.'s method for learning truth and polarity directions and chose an inefficient method for learning those directions.

Also, in hindsight, I shouldn't have simply made a direction/probe training scenario for every one of their original 
datasets. I should've recognized that some of the datasets (like those in the "real world scenarios" topic) were
tiny and meant only for testing of generalization.


## Disclaimer
Disclaimer- I did look at Marks & Tegmark's source code for linear probes to ensure that I was accurately representing
their design. Based on that, I trained the baseline linear probes without a bias/intercept term and also retrained the 
TTPD probes without a bias/intercept term.  
Ironically, it turns out that Bürger et al. did include a bias/intercept term for both their TTPD probes and their 
linear probe baseline.


# Table of Contents
1. [Truth-Falsehood Separation by Layer](#truth-falsehood-separation-by-layer)
2. [Truth-Polarity Directions](#truth-polarity-directions)
3. [TTPD Probes Results](#ttpd-probes-results)
4. [TTPD vs Linear Probes](#ttpd-vs-linear-probes)

# Truth-Falsehood Separation by Layer
As Bürger et al. found, there was one layer which was clearly overall best for truth-falsehood separation (18 in the 
case of Phi-3.5-mini).  

![](report_images/tf_sep_by_layer_graph.png)

However, I did the truth-falsehood separation analysis on all datasets rather than just 4, and I found that the 
greatest separation was found after later layers (22-29) for a small minority of datasets.  

![](report_images/tf_sep_by_dataset_and_layer_heatmap.png)

Based on this, I chose to focus on layer 18 (the global peak) and layer 25 (roughly in the middle of the later peaks) 
for learning truth and polarity directions (and the TTPD and baseline-linear probes). I also tried learning truth and 
polarity directions (and probes) for the concatenation of layers 18 and 25 activations, on the hunch (which proved 
incorrect) that there might be some information in the layer 25 activations that would be complimentary to the layer 18 
activations' information (e.g. at a different level of abstraction).

# Truth-Polarity Directions

Bürger et al. briefly say that they solve for truth and polarity directions by using ordinary least squares (with a 
closed-form solution for those directions). I misunderstood this as a result of insufficient familiarity with matrix 
calculus and statistics, so I used a least squares solver from scipy that was intended for nonlinear problems. While
I correctly implemented their definition of the loss function for learning truth and polarity directions from activation
data and truth/polarity labels, the use of a nonlinear solver rather than the closed-form solution to the 
actually-linear problem made the direction-learning process so slow that I didn't try training multiple directions and 
probes for each scenario and averaging the resulting metrics (which Bürger et al. had done in order to reduce noise in 
their findings).

I evaluated the quality of a given scenario's truth and polarity directions on their own by computing the reconstruction
loss for that scenario's training data with just the mean vector of that data and seeing what percentage of that 
reconstruction loss was eliminated by introducing the learned truth/polarity directions and that data's truth/polarity 
labels. I also evaluated how well the directions generalized to the validation-split data for the same scenario in a 
similar manner; this included contrasting the reconstruction loss reduction in the validation data when the mean vector
had been calculated from the training data vs when the mean vector had been calculated from the validation data.

The truth-falsehood separation ratio for a given layer and a given scenario (if the current scenario involved several 
datasets, the separation ratio for the scenario was the mean of the ratios its datasets) was strongly correlated with
the above metrics for evaluating learned truth and polarity directions. The Pearson correlation coefficient was ~0.83 
for layer 18 activations, ~0.84-0.85 for layer 25 activations, and ~0.85 for the concatenation of layer 18 and 25 
activations.

# TTPD Probes Results

## By Layer

The layer 18 TTPD probes were superior to the layer 25 TTPD probes (by 5.3pp of validation accuracy on 
average). They even beat them in scenarios like the disjunctive variants of animal class or element symbols, where the 
best truth-falsehood separation was much later than layer 18.

The layer 18 TTPD probes were superior to the layers-18-and-25 TTPD probes, though by somewhat less (3.8pp 
of validation accuracy on average). 

## By Scenario

Bürger et al. focused on models with 7B to 27B parameters (primarily studying Llama 3 8B) while this project tried to 
reproduce their results on a markedly smaller model (Phi-3.5-mini has 3.82B parameters).  
As a result, it is perhaps unsurprising that the TTPD probes in this project struggled much more than the Llama 3 8B
probes did on topics like facts and inventors (which required the model pretraining to have memorized relatively more 
obscure pieces of trivia or world knowledge), even just in terms of training and validation accuracy. 

It's interesting that statements involving conjunctions proved somewhat harder for the TTPD probes to learn than 
affirmative or negated statements (on average, training accuracy was lower for conjunctive data by 3pp and validation 
accuracy by 9pp, and combining negated with conjunctive data also reduced training and validation accuracy by 7 and 10pp
respectively). Meanwhile, statements involving disjunctions were even harder for the TTPD probes to learn (15-16pp
lower training or validation accuracy for that variant on its own and 11-12pp lower training or validation accuracy 
when combining affirmative and disjunctive data).

The probes trained on all 4 data variants for a given topic actually on average had training and validation accuracies 
that were almost as bad as the probes trained only on the (challenging for Phi-3.5-mini) disjunctive data.  
Meanwhile, jumping from train/validation accuracy to generalization, the "affirmative+negated+conjunction+disjunction"-data (aka "all 4 variants") scenarios on average generalized slightly better to 
unseen topics' disjunctive datasets than the just-disjunction-data scenarios.

Further details on the different TTPD probes' performance (both in terms of training/validation accuracy and in terms of generalization) can be found in [this notebook](analyzing_overall_results.ipynb)

# TTPD vs Linear Probes

While Bürger et al. also implemented Contrast-Consistent Search and Mass Mean probes as baselines, I only implemented the linear probes from [Marks & Tegmark](https://arxiv.org/abs/2310.06824) as a baseline because of limited time.

The baseline linear probes had higher validation (within topic/variant) accuracy than the TTPD probes that I trained for Phi-3.5-mini (by 10 pp on average).

Also, the baseline linear probes had higher test-set accuracy (averaged over all datasets that were completely unseen in training) than the TTPD probes that I trained for Phi-3.5-mini (by ~4 pp on average).

I'm not sure why this is, considering that Bürger et al. found their TTPD probes to be competitive with their linear probes.  
I suspect that one cause might be the suboptimal method of learning truth-polarity directions that I used. I will 
follow up on this in early 2025.














