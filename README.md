# Annotated reading list for ML theory

An annotated reference list of ML theory. I didn’t compile this list! All the credit goes to Aditi Raghunathan and her reading list for the course [Theoretical and Empirical Foundations of Modern Machine Learning (2022)](https://www.cs.cmu.edu/~aditirag/teaching/15-884F22.html). I simply generated summaries via ChatGPT-4 with the Link Reader plugin and assembled the summaries into this list. You can also download a CSV with the references to import into your citation manager.

## Generalization

**[The Tradeoffs of Large Scale Learning](https://proceedings.neurips.cc/paper/2007/file/0d3180d672e08b4c5312dcdafdf6ef36-Paper.pdf)**
Authors: Léon Bottou, Olivier Bousquet (2007)
Publication: NIPS

- The paper investigates the trade-offs in large scale learning, specifically looking at the balance between computational cost and statistical accuracy.
- It introduces a theoretical framework that allows for the analysis of the trade-offs between computation and statistics in machine learning.
- The authors argue that in large scale learning, the computational cost becomes a crucial factor, and the traditional statistical view of learning is not sufficient.
- They propose that the optimal learning strategy in such scenarios is to perform a small amount of computation on many examples rather than performing a large amount of computation on a few examples.
- The paper concludes that understanding these trade-offs can lead to more efficient learning algorithms, particularly in the context of large scale learning.

**[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)**
Authors: Jonathan Frankle, Michael Carbin (2018)
Publication: arXiv

- This paper investigates the "lottery ticket hypothesis," which posits that randomly-initialized, dense neural networks contain subnetworks ("winning tickets") that - when trained in isolation - can match the test accuracy of the original network.
- The authors provide empirical evidence supporting this hypothesis, demonstrating that such subnetworks indeed exist and can be identified through a process of iterative pruning.
- They further explore the properties of these "winning tickets," finding that they are initialized such that the initial, random weights are conducive to successful optimization.
- The paper suggests that these findings could have significant implications for neural network initialization and for understanding why large, over-parameterized networks are easier to train.
- The authors conclude by proposing future research directions, including the exploration of whether these principles apply to other forms of networks and tasks, and how these "winning tickets" can be found more efficiently.

**[Exploring Generalization in Deep Learning](https://proceedings.neurips.cc/paper/2017/file/10ce03a1ed01077e3e289f3e53c72813-Paper.pdf)**
Authors: Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals (2017)
Publication: NIPS Proceedings

- The paper investigates the generalization capabilities of deep learning models, specifically questioning the traditional view that a model's ability to generalize well is tied to its capacity (number of parameters).
- The authors demonstrate that deep learning models can perfectly fit random labels, a surprising result given that this should theoretically lead to poor generalization.
- They further show that explicit regularization techniques (like weight decay or dropout) do not significantly improve generalization performance.
- The paper concludes that these findings challenge our understanding of deep learning's generalization abilities and call for a rethinking of the theoretical foundations of deep learning.

**[The Implicit Bias of Gradient Descent on Separable Data](https://www.jmlr.org/papers/volume19/18-188/18-188.pdf)**
Authors: Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, Nathan Srebro (2018)
Publication: Journal of Machine Learning Research

- This paper investigates the implicit bias of gradient descent (GD) when applied to linearly separable data.
- The authors show that GD with small initialization and sufficiently small learning rate converges to the maximum margin separator, a result that holds true even in the presence of over-parameterization.
- They further demonstrate that this implicit bias towards maximum margin solutions is not unique to GD but is shared by other optimization algorithms.
- The paper concludes that the implicit bias of GD and other optimization algorithms plays a crucial role in the generalization ability of deep learning models, providing a fresh perspective on the behavior of these models.

## Double descent, bias-variance tradeoff, kernel methods

**[Neural Tangent Kernel: convergence and generalization in Neural Networks](https://arxiv.org/pdf/1806.07572.pdf)**
Authors: Arthur Jacot, Franck Gabriel, Clément Hongler (2018)
Publication: arXiv

- The paper introduces the concept of the Neural Tangent Kernel (NTK), a new tool to analyze the behavior of Neural Networks in the infinite width limit.
- The authors show that at initialization, neural networks in the NTK parameterization are equivalent to kernel regression with the NTK.
- The paper demonstrates that during training, the function implemented by the network evolves according to a linear differential equation, the NTK remaining constant for all time.
- The authors prove that gradient descent on neural networks follows a kernel gradient descent with respect to the NTK, with the kernel function remaining constant during training.
- The paper concludes that the NTK allows for a new understanding of the dynamics of gradient descent over neural networks, and provides a new set of tools to tackle the analysis of deep learning.

**[Benign Overfitting in Linear Regression](https://arxiv.org/pdf/1906.11300.pdf)**
Authors: Peter L. Bartlett, Philip M. Long, Gábor Lugosi, Alexander Tsigler (2020)
Publication: arXiv

- The paper investigates the phenomenon of benign overfitting, where deep neural networks predict well even with a perfect fit to noisy training data, in the context of linear regression.
- The authors provide a characterization of linear regression problems for which the minimum norm interpolating prediction rule has near-optimal prediction accuracy.
- The paper shows that overparameterization is essential for benign overfitting in this setting: the number of directions in parameter space that are unimportant for prediction must significantly exceed the sample size.
- The authors find that the accuracy of the minimum norm interpolating prediction rule approaches the best possible accuracy for a much narrower range of properties of the data distribution when the data lies in an infinite dimensional space versus when the data lies in a finite dimensional space whose dimension grows faster than the sample size.
- The paper concludes that understanding the performance of prediction rules that fit the training data perfectly is a central challenge to arrive at a scientific understanding of the success of deep learning methods.

## Robustness

**[A universal law of robustness via isoperimetry](https://arxiv.org/pdf/2105.12806.pdf)**
Authors: Daniel M. Roy, Ilya O. Ryzhov (2022)
Publication: arXiv

- The paper investigates the concept of robustness in machine learning models, particularly focusing on the mathematical principles that govern this property.
- The authors propose a universal law of robustness, which they derive from the isoperimetric inequality, a fundamental concept in geometry and analysis.
- The law provides a bound on the robustness of a model, given the model's accuracy and the complexity of the classification problem.
- The authors argue that this law applies to all models and datasets, regardless of their specific characteristics.
- The paper concludes that the trade-off between accuracy and robustness is a fundamental aspect of machine learning, and that improving one often comes at the expense of the other.

**[Adversarial Examples are not Bugs, they are Features](https://proceedings.neurips.cc/paper/2019/file/e2c420d928d4bf8ce0ff2ec19b371514-Paper.pdf)**
Authors: Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, Aleksander Madry (2019)
Publication: NeurIPS

- The paper explores the phenomenon of adversarial examples in machine learning, arguing that these are not bugs but rather features of the model.
- The authors propose that adversarial examples can be attributed to the presence of non-robust features in the data, which are highly predictive but also brittle and incomprehensible to humans.
- They demonstrate that these non-robust features are widespread in standard datasets and can lead to adversarial perturbations.
- The paper also discusses adversarial transferability, suggesting that since any two models are likely to learn similar non-robust features, perturbations that manipulate such features will apply to both.
- The authors conclude that adversarial vulnerability is a human-centric phenomenon, and that non-robust features can be as important as robust ones from the perspective of standard supervised learning.

**[Understanding the failure modes of out-of-distribution generalization](https://arxiv.org/pdf/2010.15775.pdf)**
Authors: Vaishaal Shankar, Alex Fang, Wenshuo Guo, Sara Fridovich-Keil, Ludwig Schmidt, Jonathan Ragan-Kelley, Benjamin Recht, Moritz Hardt, John Miller, Ludwig Schmidt (2022)
Publication: arXiv

- This paper investigates the failure modes of out-of-distribution (OOD) generalization in machine learning models. It specifically focuses on the brittleness of models when faced with distribution shifts, which is a crucial issue for the reliability of machine learning systems.
- The authors propose a new framework, the "OOD generalization triangle", to understand the relationship between in-distribution generalization, OOD generalization, and distribution shift.
- The paper shows that the OOD generalization error can be decomposed into three components: in-distribution error, shift magnitude, and shift direction. This decomposition helps in understanding the failure modes of OOD generalization.
- The authors demonstrate that the shift direction, which is often ignored in the existing literature, plays a crucial role in determining the OOD generalization error.
- The paper concludes that understanding the shift direction is key to improving the robustness of machine learning models against distribution shifts.

**[Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization](https://arxiv.org/pdf/2107.04649.pdf)**
Authors: John Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, Ludwig Schmidt (2022)
Publication: arXiv

- This paper empirically demonstrates a strong correlation between in-distribution and out-of-distribution performance for a wide range of models and distribution shifts.
- The authors show that this correlation holds across model architectures, hyperparameters, training set size, and training duration, and is more precise than what is expected from existing domain adaptation theory.
- The paper also investigates cases where the correlation is weaker, such as some synthetic distribution shifts from CIFAR-10-C and the tissue classification dataset Camelyon17-WILDS.
- The authors provide a candidate theory based on a Gaussian data model that shows how changes in the data covariance arising from distribution shift can affect the observed correlations.
- The paper concludes that improving in-distribution performance reliably improves out-of-distribution performance. However, it is currently unclear whether improving in-distribution performance is the only way, or even the best way, to improve out-of-distribution performance.

## Causality

**[On causal and anti-causal learning](https://icml.cc/2012/papers/625.pdf)**
Authors: Bernhard Schölkopf, Dominik Janzing, Jonas Peters, Eleni Sgouritsa, Kun Zhang, and Joris Mooij (2012)
Publication: ICML 2012

- The paper investigates the difference between causal and anti-causal learning, where causal learning predicts effects from causes, and anti-causal learning predicts causes from effects.
- The authors argue that causal learning is fundamentally easier than anti-causal learning. This is because the causal direction is independent of the underlying distribution, while the anti-causal direction is not.
- The paper introduces a new algorithm for causal feature selection, which is based on the idea that causal features are easier to predict than anti-causal features.
- The authors demonstrate the effectiveness of their algorithm through a series of experiments on synthetic and real-world data.
- The paper concludes that understanding the difference between causal and anti-causal learning can lead to more effective machine learning algorithms.

**[Invariant Risk Minimization](https://arxiv.org/pdf/1907.02893.pdf)**
Authors: Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, David Lopez-Paz (2020)
Publication: Arxiv

- The paper addresses the fundamental problem in machine learning where machines inherit biases from the data they are trained on, leading to spurious correlations and poor generalization to new test distributions.
- The authors propose Invariant Risk Minimization (IRM), a novel learning paradigm that estimates nonlinear, invariant, causal predictors from multiple training environments, enabling out-of-distribution (OOD) generalization.
- The paper presents a mathematical formulation of IRM and discusses its implementation details, including how to estimate the objective using mini-batches for stochastic gradient descent.
- The authors also explore the relationship between invariance, causality, and OOD generalization, arguing that invariant predictors can be written as linear data representations of different ranks.
- The paper concludes with a discussion on future research directions, including the benefits of enforcing non-linear invariances and constructing invariance penalties for non-linear invariances.

## Unsupervised learning

**[Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/pdf/1804.09170.pdf)**
Authors: Oliver, A., Odena, A., Raffel, C., Cubuk, E. D., & Goodfellow, I. (2018)
Publication: arXiv preprint arXiv:1804.09170

- The paper investigates the performance of deep semi-supervised learning (SSL) algorithms under realistic conditions. It argues that previous evaluations of these algorithms may have been overly optimistic due to certain experimental design choices.
- The authors propose a new evaluation methodology that includes factors such as the presence of out-of-distribution examples in the unlabeled dataset, the use of data augmentation, and the variability in performance due to different model initializations and architectures.
- The paper finds that under these more realistic conditions, the performance of deep SSL algorithms is significantly worse than previously reported. In particular, the authors find that the presence of out-of-distribution examples in the unlabeled dataset can severely degrade performance.
- The authors also find that the choice of data augmentation strategy and the variability in performance due to different model initializations and architectures can significantly impact the performance of deep SSL algorithms.
- The paper concludes by calling for more rigorous evaluation methodologies in future SSL research to ensure that the reported performance of these algorithms is representative of their performance under realistic conditions.

**[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)**
Authors: He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021)
Publication: arXiv preprint arXiv:2111.06377

- The paper presents a novel approach to self-supervised learning for computer vision tasks, called Masked Autoencoders (MAE), which is based on masking random patches of the input image and reconstructing the missing pixels.
- The authors propose an asymmetric encoder-decoder architecture, where the encoder operates only on the visible subset of patches, and a lightweight decoder reconstructs the original image from the latent representation and mask tokens.
- The paper finds that masking a high proportion of the input image (e.g., 75%) yields a nontrivial and meaningful self-supervisory task, which enables efficient and effective training of large models.
- The authors demonstrate that their approach can achieve high accuracy (87.8%) on the ImageNet-1K dataset, outperforming previous methods that use only ImageNet-1K data.
- The paper concludes that the MAE approach allows for learning high-capacity models that generalize well, and shows promising scaling behavior in downstream tasks.

**[Emerging properties in self-supervised vision transformers](https://arxiv.org/pdf/2104.14294.pdf)**
Authors: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou (2022)
Publication: arXiv

- The paper investigates the properties of self-supervised learning in Vision Transformers (ViTs) and how they compare to Convolutional Neural Networks (CNNs).
- The authors focus on the question of whether the self-attention mechanism in ViTs can learn to localize objects in an image without explicit supervision, a property known as "objectness."
- The paper finds that ViTs trained with self-supervised learning can indeed learn to localize objects, and that this ability improves with the scale of the model and the amount of training data.
- The authors also find that the self-attention maps of ViTs can be interpreted as object saliency maps, providing a form of interpretability for these models.
- The paper concludes that self-supervised learning can be a powerful tool for training ViTs, and that these models have promising properties for tasks such as object detection and image segmentation.

**[Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss](https://arxiv.org/pdf/2106.04156.pdf)**
Authors: Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma (2022)
Publication: arXiv

- This paper presents a theoretical framework for self-supervised learning without requiring conditional independence, which is a common assumption in previous works.
- The authors introduce a novel concept of the augmentation graph on data, where edges connect augmentations of the same datapoint, and ground-truth classes naturally form connected sub-graphs.
- They propose a loss function, the spectral contrastive loss, that performs spectral decomposition on the population augmentation graph and can be succinctly written as a contrastive learning objective on neural net representations.
- The paper proves that, under a simple and realistic data assumption, linear classification using representations learned on a polynomial number of unlabeled data samples can recover the ground-truth labels of the data with high accuracy.
- Empirically, the features learned by the proposed objective can match or outperform several strong baselines on benchmark vision datasets.

## Distribution shifts

**[Domain-adversarial training of neural networks](https://arxiv.org/pdf/1505.07818.pdf)**
Authors: Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016)
Publication: The Journal of Machine Learning Research

- The paper introduces a new approach for domain adaptation in deep networks, called Domain-Adversarial Neural Network (DANN). The approach aims to learn a feature representation that is useful for the learning task and is also invariant to the change of domains.
- The authors propose a new regularization approach that encourages the learned features to be domain-invariant. This is achieved by adding a domain classifier to the network and training it adversarially against the feature learner.
- The paper demonstrates that the proposed approach can significantly reduce the error rate in domain adaptation tasks. The authors show that DANN outperforms standard neural networks and other domain adaptation methods on several benchmark datasets.
- The authors also provide a theoretical analysis of their method, showing that the domain-adversarial training process can be interpreted as minimizing a certain upper bound of the expected risk on the target domain.
- The paper concludes that the proposed DANN approach is a promising direction for domain adaptation in deep networks. The authors suggest that future work could explore other types of domain-invariant representations and investigate the use of DANN in other types of learning tasks.

**[Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://arxiv.org/pdf/1909.13231.pdf)**
Authors: Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020)
Publication: Proceedings of the 37th International Conference on Machine Learning

- The paper proposes Test-Time Training (TTT), a method for improving the performance of predictive models when training and test data come from different distributions. The approach turns a single unlabeled test sample into a self-supervised learning problem, updating the model parameters before making a prediction.
- The authors argue that supervised learning struggles with generalization under distribution shifts. They propose to learn from these shifts at test time, allowing the model parameters to depend on the test sample but not its unknown label.
- The TTT method creates a self-supervised learning problem based on a single test sample, updating the model parameters at test time before making a prediction. The authors use the task of rotating each input image by a multiple of 90 degrees and predicting its angle as an auxiliary task.
- The paper demonstrates that TTT leads to improvements on diverse image classification benchmarks aimed at evaluating robustness to distribution shifts. The authors show that their algorithm makes substantial improvements under distribution shifts, while maintaining the same performance on the original distribution.
- The authors conclude that TTT is a promising approach for dealing with distribution shifts in predictive models. They suggest that future work could explore other types of self-supervised tasks and investigate the use of TTT in other types of learning tasks.

## Foundation models

**[Model-agnostic meta-learning for fast adaptation of deep networks](https://arxiv.org/pdf/1703.03400.pdf)**
Authors: Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
Publication: Proceedings of the 34th International Conference on Machine Learning

- The paper introduces a method called Model-Agnostic Meta-Learning (MAML) that is designed to help deep learning models adapt quickly to new tasks.
- The key idea of MAML is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of gradient steps.
- The authors demonstrate that MAML is applicable to any model trained with gradient descent and to any machine learning problem that can be cast as learning a function, including classification, regression, and reinforcement learning problems.
- The experiments show that the approach is effective for few-shot learning in the context of image recognition and reinforcement learning tasks.
- The paper concludes that MAML provides a promising approach for few-shot learning and rapid adaptation to new tasks, but also notes that there are many interesting directions for future work, including exploring different types of prior knowledge and meta-objectives.

**[The power of scale for parameter-efficient prompt tuning](https://arxiv.org/pdf/2104.08691.pdf)**
Authors: Brian Lester, Rami Al-Rfou, Noah Constant (2021)
Publication: arXiv preprint

- This paper explores "prompt tuning," a mechanism for learning "soft prompts" to condition frozen language models to perform specific downstream tasks.
- The authors show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, their method matches the strong performance of model tuning (where all model weights are tuned).
- The paper demonstrates that their end-to-end learned approach outperforms GPT-3’s few-shot learning by a large margin.
- The authors also show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer and enables efficient "prompt ensembling."
- The paper concludes that prompt tuning is a promising method for adapting large language models, offering a balance between performance and efficiency, and that it opens up several avenues for future research.

**[Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361.pdf)**
Authors: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei (2020)
Publication: arXiv

- The paper investigates the relationship between the performance of neural language models and their scale, in terms of model size, dataset size, and the amount of computation.
- The authors find that, in many cases, increasing these factors leads to improved performance, even beyond the scales that are currently common in the field.
- They also find that the benefits of scale are not constant, but rather exhibit "power-law" scaling, meaning that the benefits decrease as scale increases, but do not disappear entirely.
- The authors suggest that these findings could have significant implications for the future of AI research, as they suggest that simply scaling up existing models and techniques could lead to continued improvements in performance.
- However, they also note that this approach could have significant costs, both in terms of the computational resources required and the potential environmental impact.

**[What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://arxiv.org/pdf/2208.01066.pdf)**
Authors: Shivam Garg, Dimitris Tsipras, Percy Liang, Gregory Valiant (2023)
Publication: arXiv

- The paper explores the concept of in-context learning, where a model learns to generate outputs based on a sequence of input-output pairs, without any parameter updates.
- The authors focus on the ability of Transformer models to perform in-context learning of simple function classes, such as linear functions, sparse linear functions, two-layer neural networks, and decision trees.
- They find that Transformers can be trained to perform in-context learning of these function classes, with performance comparable to or exceeding that of task-specific learning algorithms.
- The authors also find that the performance of the trained models is robust to distribution shifts between the training data and inference-time prompts, as well as between the in-context examples and the query input during inference.
- The study suggests that Transformers can encode complex learning algorithms in a single forward pass, and that increasing the model's capacity can significantly improve its performance.

## Benchmarking LLMs

**[Beyond the imitation game: quantifying and extrapolating the capabilities of language models](https://arxiv.org/pdf/2206.04615.pdf)**
Authors: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei (2022)
Publication: arXiv

- The paper investigates the capabilities of large language models, specifically focusing on GPT-3, and proposes a new methodology to quantify and extrapolate their performance.
- The authors introduce a new measure, the "pseudo-perplexity", to quantify the performance of language models. This measure is based on the model's ability to predict held-out human text.
- The paper explores the relationship between model size and performance, finding that performance continues to improve with increasing model size, albeit with diminishing returns.
- The authors also investigate the model's ability to generalize from the training data and perform tasks that are not explicitly present in the training data.
- The paper concludes that while large language models like GPT-3 have impressive capabilities, there are still many tasks where they fall short, indicating the need for further research and development in this field.

**[On the dangers of stochastic parrots: can language models be too big?](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)**
Authors: Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell
Publication: [FAccT '21](https://dl.acm.org/doi/proceedings/10.1145/3442188)

- The paper critically examines the trend in Natural Language Processing (NLP) of developing and deploying increasingly larger language models, such as BERT, GPT-2/3, and Switch-C. These models have advanced the state-of-the-art on many tasks, largely through the methodology of pretraining on large datasets and fine-tuning for specific tasks.
- The authors pose the question: "How big is too big?" They explore the potential risks associated with these large language models, including environmental, financial, and ethical considerations.
- The paper highlights the environmental and financial costs of training large language models. It suggests that these costs should be carefully weighed before deciding to develop such models.
- The authors recommend investing resources into curating and carefully documenting datasets, rather than indiscriminately ingesting all available web data. This approach could help to mitigate some of the risks associated with large language models.
- The paper encourages pre-development exercises to evaluate how the planned approach aligns with research and development goals and supports stakeholder values. It also advocates for exploring research directions beyond simply creating larger and larger language models.

# Prompt

```
Fetch the following papers. Based on their abstracts and the content in the introduction, write 3-5 key points about these papers. Make sure to highlight the key questions investigated by the paper, and the core conclusions. Target a mathematically knowledgeable professor of neuroscience who is outside of this field. Write the results in the format:

[Paper 1 name](https://link)
Authors (date)
Publication name

* point 1
* Point 2
* Point 3, etc.

[Paper 2 name]...
```
