---
layout: post
title: [Slim but Smart -- a Review on ALBERT]
authors: Anonymous
<!-- authors: Elaba, Kevin, Carnegie Mellon University; Gadiyar, Aditya, Carnegie Mellon University; Wang, Yiqing, Carnegie Mellon University -->
tags: [natural_lanugage_inference, NLP, deep_learning, model_reduction]
---

What does it take to build an intelligent robot that comprehends text and responds appropriately to relevant questions? For example, how good can a robot be at taking the [English reading comprehension exams (RACE)](https://arxiv.org/pdf/1704.04683.pdf) administered to Chinese middle-schoolers and high-schoolers? In 2018, Google’s AI Language team published [BERT](https://www.marketingaiinstitute.com/blog/bert-google), a breakthrough in natural language processing designed to answer this very question. Named for its bidirectional transformer structure, BERT established the significant performance advantages of pre-trained models for learning tasks; the model utilized **340 million** parameters to achieve an [accuracy of 72%](https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf). Even with Tensor Processing Units (TPUs) – Google’s proprietary hardware developed specifically for large-scale deep learning tasks – BERT required [four days of training on 64 TPUs](https://huggingface.co/blog/bert-101)!\
\
![race]({{ https://arxiv.org/pdf/1704.04683.pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/race.png)
<p align='center'><sup>Here's an example question taken from the RACE test (not the clearest abbreviations, I know). If you have a second to scan through the story above, you will realize that it is pretty simple for humans, but it takes a lot of compute for a computer to understand the passage.</sup></p>\
As with any machine learning breakthrough, there are two questions that immediately come to mind. Can we achieve the same results more efficiently? And can we improve upon this performance?\
\
Let’s start out with the first question: we need a way to slim down this gigantic model! At a basic level, memory limitations are an immediate bottleneck. Even with dedicated hardware, there is always an incentive to shrink our models down to reduce the memory demands of training. Additionally, large models like BERT are typically trained in a distributed manner across many machines; the communication overhead associated with such an approach is also significant. Ideally, we can address both these issues by shrinking the number of parameters in the model.\
\
Enter ALBERT – short for A Lite BERT. ALBERT is a smaller version of BERT that strives to circumvent these exact limitations while also making performance advancements.
## Improvements Offered by ALBERT
ALBERT utilizes three key features to be slim but smart. The first two are techniques used to slim down the model and the last one boosts its performance. 
1. [Factorized embedding parameterization](#factorized-embedding-parameterization)
2. [Cross-layer parameter sharing](#croos-layer-parameter-sharing)
3. [Self-supervised loss for sentence-order prediction](#sentence-order-prediction-loss).
<p></p>
### Factorized Embedding Parameterization
Machines use numerical values, but words are categorical. The way we can translate words to trainable metrics is by using word embeddings, which are a set of numerical values unique to each word or word-related token (such as WordPieces embeddings used in [BERT](https://arxiv.org/pdf/1609.08144.pdf)). These values can be tweaked after being propagated across multiple blocks of the transformer to reflect higher level relationships between words.\
![bert]({{ https://arxiv.org/pdf/1609.08144.pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/bert.png)
<p align='center'><sup>The architecture of BERT, showing <em>E</em> and <em>T</em>.<br> (Annotations added by authors of this blog)</sup></p>
In the original BERT model, the input word embeddings *E* have the same size as the hidden word embeddings *T*. However, ALBERT separates the two embedding matrices. This is because WordPiece (input) embedding E are designed to learn context-independent representations, whereas hidden-layer embeddings T are meant to learn context-dependent representations, and BERT performs well precisely because of these context-dependent representations. ALBERT delinks the size of *E* vectors from that of *T*. It projects the input through a lower dimensional space before projecting it into the hidden space. This change led to a significant decrease in parameters. 
### Cross-layer Parameter Sharing
In a transformer layer, there is an attention network and a feed-forward network. In the original BERT model, the multi-head self-attention segment and feed-forward segment each have their own parameters. ALBERT instead has parameters that are shared between these subsegments. By sharing parameters across all layers, the total number of parameters within ALBERT is greatly reduced relative to BERT. In addition to decreasing parameter redundancy and improving overall model efficiency, this feature also stabilizes the parameters such that the model is more generalizable.
<div class='message'>
<strong>Sidenote</strong>: stabilizing the parameters means that the difference between the inputs and outputs of different layers are not widely different. This can be analyzed using the L2 distances and cosine similarity of the input and output embeddings of each layer. As shown here, ALBERT’s curve is smooth, suggesting more stable parameters.
<img src='/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/cross.png'>
</div>
### Sentence Order Prediction Loss
BERT used Next-Sentence Prediction (NSP) as a loss function, which predicts “yes” or “no” as to whether two segments of text appear consecutively in the original text. The NSP’s objective was designed to improve performance on other tasks, such as understanding whether two sentences have the same meaning, but was later deemed ineffective. This might be due to the fact that NSP tries to combine the prediction of topics and the coherence of the sentences into one task. Insead, ALBERT uses a new loss called Sentence Order Prediction (SOP), which only focuses on modeling inter-sentence coherence. This has helped ALBERT achieve better results.
<p></p>
## ALBERT's Experimental Setup
For comparison, the authors preprocessed their data and performed training with the same methodology that was used for BERT. ALBERT was trained on the English Wikipedia and BOOKCORPUS datasets, which together contain ~16 GB of uncompressed data. ALBERT also uses a vocabulary size of 30,000 and groups together the vocabulary in groups of 1-grams, 2-grams, or 3-grams.
<div class="message">
<strong>Side note</strong>: contiguous sequences of n words can be grouped together into n-grams. 
</div>
<p></p>
## Impacts of ALBERT's Improvements
Let’s take a deeper dive into just how much better ALBERT is compared to its bigger brother.
### High level Architecture/Parameter Comparison
ALBERT is released in 4 different model sizes, with all versions being a fraction of the size of BERT. For example, ALBERT-base has 9x fewer parameters than BERT-base, and ALBERT-large has about 18x fewer parameters compared to BERT-large.
![param]({{ https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/param.png)
### Overall Improvement
The ALBERT-xxlarge model performs significantly better than BERT-large while having 70% fewer parameters! Additionally, the biggest BERT model performed significantly worse than BERT-base on all metrics, which indicates that **bigger is not always better**. After training for roughly the same amount of time, ALBERT-xxlarge is significantly better than BERT-large: +1.5% better on average, with the difference on the RACE dataset as high as +5.2%.\
\
![overall]({{ https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/overall.png)
<p align='center'><sup>The results of ALBERT’s performance on a variety of benchmark datasets, including the Reading Comprehension test (RACE) mentioned before</sup></p>
The experimental data demonstrates that ALBERT makes notable performance gains over BERT while having considerably fewer parameters. Let’s now assess the individual impact of each of ALBERT’s key design choices:
### Factorized Embeddings Parameterization
In order to assess the impact of the factorized embeddings, four different vocabulary embedding sizes E were used. For each embedding E, the score was collected for the different downstream tasks. This experiment was performed twice: once for BERT-style models with no parameter sharing and once with ALBERT-style models featuring parameter sharing. For the BERT-style models, larger embedding sizes correlated to superior performance, although the performance difference was relatively small. For the ALBERT-style model, it was found that an embedding size of 128 consistently yielded the best performance. Therefore, the authors decided to use an embedding size of 128 for all future ALBERT experiments/settings.\
\
![factor]({{ https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/factor.png)
### Cross-Layer Parameter Sharing
To assess the cross-layer parameter sharing, four types of parameter-sharing were assessed on the downstream tasks: not-shared (BERT-style), all-shared (ALBERT-style), only sharing attention parameters, and only sharing feed-forward parameters. The latter two are essentially intermediate strategies where only some parameters are shared. All four strategies were trialed with an embedding size of 768 and 128. It was found that the all-shared strategy employed by ALBERT actually impairs performance for both embedding sizes. However, in comparing the attention-only sharing with the feed-forward-only sharing, it appears that the feed-forward sharing contributes much more to the overall performance drop seen by ALBERT’s all-shared approach. Despite these results, it was found that for the chosen embedding size setting of 128, ALBERT’s performance deficit is relatively minimal.\
\
![cross2]({{ https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/cross2.png)
### Sentence Order Prediction
To assess the effectiveness of sentence order prediction, SOP was compared against NSP (BERT-style) as well as the case where no inter-sentence loss is used. The scores were collected for each of these three loss strategies across the different downstream tasks. The results demonstrate that NSP loss doesn’t aid in the SOP task, indicating that NSP only models topic shifts but not inter-sentence coherence. Conversely, the findings show that SOP loss performs well for both tasks. Most importantly, SOP loss consistently improves performance for multi-sentence encoding tasks.\
\
![sop]({{ https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf }}/public/images/2021-12-01-slim-but-smart-a-review-on-ALBERT/sop.png)
<p></p>
## Final Words
Model pruning is commonly used to reduce the size of complex models for faster inference, but we can also save training time (along with inference time) and compute resources by building a smaller model from the very beginning. ALBERT did this through making the vocabulary embedding layer smaller, sharing parameters across all of its layers, and using a loss function better suited for the task of comprehending sentence coherence. The key takeaway from ALBERT is that <span style='color:#CC2F67'>larger models are not always better! Clever design choices can make the models smaller but perform better. </span>