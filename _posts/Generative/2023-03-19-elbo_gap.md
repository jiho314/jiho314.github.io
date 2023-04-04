---
title: "The Gap between ELBO and actual implementation"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
- Generative
---

## The Gap between ELBO and actual Implementation

Evidence Lower BOund
$$
\begin{align}
 \log p_\theta(x) &= \int \log p_\theta(x,z)dz \\
&= \int \log \frac {p_\theta(x,z) q_{\phi}(z|x)}{q_{\phi}(z|x)}dz \\
& \geq \log E_{q_{\phi}(z|x)}[\log \frac{p_\theta(x,z)}{q_{\phi}(z|x)}] = \text{ELBO}
\end{align}
$$



Dividing ELBO into 'reconstruction term' and 'prior matching term'. 
$$
\begin{align}\text{ELBO} &= E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{q_{\phi}(z|x)} ]\\
		&= E_{q_{\phi}(z|x)}[\log p_\theta(x|z)] -E_{q_{\phi}(z|x)}[\log \frac{p(z)}{q_{\phi}(z|x)}]\\
		&= \underbrace{E_{q_{\phi}(z|x)}[\log p_\theta(\hat{x}|z)]}_{\text{reconstruction term}} - \underbrace{D_{KL}(q_\phi(\hat{z}|x)||p(z))}_{\text{prior matching term}}
\end{align}
$$


Reconstruction term의 수식 그대로의 의미는,  
**'실제 $x$의 distribution $p$' 에 대한 '만들어진 $\hat{x}$ data들'의 likelihood** 이다.   
Prior matching term은 $z$에 대한 실제 prior와 $x$로부터 만들어진 $\hat{z}$ 의 distribution 차이를 의미한다. 

여기서 VAE 는 $p(z)=N(z;0,I)$ , z가 Gaussian을 따른다는 가정하에 '$-\text{ELBO}$'식을 Loss term으로 집어 넣는다고 한다. 하지만 VAE에서 사용되는 model은 두 개의 neural net( $\theta, \phi$) 뿐이다. 각각을 통해 $x, \space z$를 각각 생성할 수는 있지만, 그것들의 distribution을 어떻게 계산한다는 것인가?

 실질적으로 VAE에 적용되는 loss function은 아래와 같다. distribution에 대한 likelihood 를 의미하는 term은 단순히 실제 $x$ 와 reconstruct 된 $\hat{x}$ 간의 mse loss나 binary cross entropy loss로 치환되었다.

```python
def loss_function(recon_x, x, mu, logvar):
    reconstruction = F.mse_loss(recon_x,x)
    #reconstruction = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    prior_matching = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction + prior_matching
```

<br/>

이러한 이론적인 ELBO 와 실질적 Loss term간의 괴리는 그 사이에 추가된 가정들에 의한 것이었다.    
VAE에서는 ELBO에서 생성된 데이터들이 Gaussian / Bernoulli 라는 가정하에 ELBO를 계산해낸 것이다. 각각의 가정하에 Likelihood를 Maximize 하는 것은 각 MSE / BCE 로 최적화하는 것과 동치이다. (Gaussian 분포의 데이터를 MLE 하는 것 = 각각의 데이터로 MSE 하는 것)

Prior matching term 역시, 두  Gaussian 분포의 MLE는, 그 평균과 분산에 의해 계산된다는 결론 하에 구현된 것이다.  VAE의 encoding 과정에서($\phi: x \rightarrow z$ ),  z vector를 직접 만들지 않고 평균과 분산만 출력하는 것도 이러한 맥락에서 납득할 수 있는 것이다.

ELBO 역시 실제 $p(z|x)$를 추정하기 위해 $q_\phi(z|x)$를 도입하여 만들어졌지만, ELBO 식 그대로를 계산을 할 수 없는 것이다. 따라서 ELBO를 계산해내기 위해서는 필연적으로 추가적인 가정(inductive bias)이 필요하고, 어떠한 가정을 할 것이냐 에 따라 아예 다른 모델이 될 수 있다. Generative 분야를 압도하는 Diffusion 역시, Hierarchical(계층적)-VAE 구조에 다른 가정을 적용하여 ELBO를 계산해낸 것으로 볼 수 있다.

 참고논문: [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)



### Discussion

VAE에 대한 공부가 부족하여 각 notation의 수학적 의미를 정확히는 이해하지 못하였다. p는 실제 distribution을 의미하는 notation이고, q는 추정하는 notation임을 전제하고, Discussion을 진행하였다.

최초 ELBO의 유도된 식과 VAE에서 사용되는 식에서 notation간의 괴리가 있었다. 많은 ELBO 유도 과정에서, $p(x)$는 input event $x$ 가 고정이면 $p(x)$ 또한 고정된 값으로 이미 존재하는 $x$의 distribution을 의미한다고 말한다. 그에 따라 ELBO를 최대화 하는 것은, $D_{KL}(q_\phi(z|x)|p(z|x))$를 최소화함을 의미한다고 하는 것이 ELBO의 핵심이다. 하지만 VAE에서는 generative model이 도입되어 $p_\theta(x)$ 가 등장하여, $p(x)$ term이 고정된다는 의미가 사라진다고 볼 수 도 있을 것 같다. 이 두 논리 간의 괴리를 해결하기 위해서, vae에 대한 다른 논문들을 더 읽어보아야 할 것 같다.







<!--ELBO를 최대화 함으로서 $p_\theta(x)$ 를 최대화 한다는 것은,  $\theta$ 로 생성된 data의 distribution   -->

