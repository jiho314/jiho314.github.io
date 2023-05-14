---
title: "Maximizing ELBO의 posterior 근사 의의는 VAE에서 사라지는가?"
toc: true
toc_sticky: true
toc_lable: "Main Contents"
use_math: true
categories:
- Generative
---

## Maximizing ELBO의 posterior 근사 의의는 VAE에서 사라지는가?

Maximizing ELBO의 의의는 크게 ELBO 내부에서와, ELBO가 유도된 전체 식에서로 나누어 볼 수 있다. ELBO 내부의 reconstruction term 과 prior matching term은 VAE를 공부한 사람이라면 모두 알 것이다. 이 글은, ELBO를 포함한 전체 식에서 일어나는 일에 대한 것이다. 이와 관련해 공부하다가 상충된 두 개의 논리가 등장해 고민해 보았다.

### Maximizing ELBO: Original

ELBO의 원래 의의는 posterior를 근사하는데 있다.  
(True distribution $p$에 대해서, posterior가  $p(z|x)$ 인 상황)   
$p(z|x)$가 intractable하여 $q_\phi(z|x)$를 설정하고, 두 distribution의 차이를 아래와 같이 수식전개를 한 것이다.
$$ {equation}
\begin{align}
D_{KL}(q_\phi(z|x)||p(z|x)) &= E_{q_\phi(z|x)}[\log \frac{q_\phi(z|x)}{p(z|x)}] 
\newline
&= -\text{ELBO} + E_{q_\phi(z|x)}[\log p(x)]
	\newline
	&= -\text{ELBO} + \log p(x)
\end{align}
$$
위의 식을 $p(x)$를 기준으로 정리하면 아래와 같다. 중요한 것은 $p(x)$는 고정된 사건(데이터) $x$ 에 대해서 고정된 확률 값을 갖는다는 것이다.

따라서 ELBO를 maximizing 하는 것은 KL-Divergence term을 최소화 함으로서 원래 목적이었던 posterior 근사를 가능하게 해준다. 하나 기억해야할 것은 KL-D term이 최소화 되는 것은 $p(x)$가 고정이었기 때문에 가능한 것이다.
$$
\begin{equation}
\log p(x) = \text{ELBO} + D_{KL}(q_{\phi}(z|x)||p(z|x))
\end{equation}
$$




### Maximizing ELBO: in VAE

하지만 VAE에서는 전혀 다른 논리를 가져온다. Generation을 위한 $\theta$가 도입되면서 좌변은, $p_\theta$($\theta$가 만들어내는 distribution)에 대한 train data $x$ 의 확률을 의미하게 되었다. 따라서 좌변이 maximize 될수록 $\theta$ 가 생성하는 data의 distribution은 train data $x$ 에 가까워질 것이다.  그렇다면 간단하게 ELBO를 maximize하면 $\log p_\theta(x)$ 역시 Maximize 될 것이고, 이것이 VAE에서 말하는 Maximizing ELBO의 의의이다.





$$
\begin{equation}
\log p_\theta(x) \geq \text{ELBO}
\end{equation}
$$





여기서 의문이다.

더 이상 $p_\theta(x)$가 고정된 값이 아니라면, ELBO를 최대화 하는 것으로 KL-term을 최소화 할 수 없으므로, posterior 근사는 이루어지지 않는 것인가?



## Does VAE Approximate the Posterior?

생각해보면 좌변이 고정된 값이 아니라면 ELBO를 아무리 최대화 해도 KL-D term이 최소화된다는 보장은 어디에도 없다. 그렇다면 VAE에서는 posterior가 근사되지 않는 것인가? 


$$
\begin{equation}
\log p_\theta(x) = \underbrace{E_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction term}} + \underbrace{D_{KL}(q_\phi(z|x)||p(z))}_{\text{prior matching term}}+ \underbrace{D_{KL}(q_\phi(z|x)||p_\theta(z|x))}_{\text{"The" Posterior Estimation}}
\end{equation}
$$


최초 ELBO는 $\phi$에 대해서만 ELBO를 maximize하지만, VAE에서는 $\theta$와 $\phi$ 모두 optimize한다. 따라서 Optimize관점을 나누어, 각각의 optimizing 방향을 생각해 볼 필요가 있다. 일단 $\theta$와 $\phi$  모두 좌변을 최대화하는 방향으로 학습되는 것은 동일하다. 그렇다면 $\theta$를 고정되어있다고 보고, $\phi$에 대해서 생각을 해보자. 

$\phi$ Optimization의 관점에서는, 좌변은 고정이므로 $q_\phi$가 posterior로 근사된다. 하지만 여기서 근사하고자 하는 distribution이 true distribution이 아닌 $\theta $에 대한 posterior $p_\theta(z|x)$ 이다. 따라서 수렴하기 직전까지는, $\phi$는 엉뚱한 posterior에 근사된다. 하지만 근본적으로, $\phi$ 의 optimize 방향은, $\theta $의 posterior 인 것은 알 수 있다.

전체 Optimization 과정에서 보면, $\phi$는 step마다 다른 posterior로 근사되고 있다. 따라서 ELBO를 Maximize 하는 과정 자체가 Posterior를 직접적으로 근사한다고는 할 수 없다. 하지만, Global minimum 지점에서만큼은 의도했던 posterior가 근사 될 것이다. 결론적으로, 기존 ELBO 최대화의 의미를 VAE 역시 가지고 있다고 볼 수 있다.

