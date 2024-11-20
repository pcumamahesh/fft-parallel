
#### Group Members
Naren Kumar Sai ($220001049$)
P C Uma Mahesh ($220001052$)

# Parallel Computation of the Fast Fourier Transform using OpenMP

### Introduction

Efficient polynomial multiplication is a fundamental task in computational mathematics, with applications ranging from signal processing to cryptography. The Fast Fourier Transform (FFT) provides an optimized approach for multiplying polynomials by converting multiplication into an interpolation problem. This project aims to use parallel computing techniques, specifically OpenMP, to accelerate the FFT and polynomial multiplication processes, exploring the performance gains from parallelizing the algorithm on multi-core architectures.

### Problem Statement

We focus on the following important use case of the FFT algorithm.
###### Polynomial Multiplication
Given coefficients of two polynomials, return the coefficients of the polynomial obtained by multiplying the given polynomials. 

### Proposed Solution 

##### Theoretical Background
Consider two polynomials of degree $n$, $a$ and $b$, where:
$$
a(x) = a_0+a_1x+a_2x^2+ \dots +a_nx^n
$$
$$
b(x) = b_0+b_1x+b_2x^2+ \dots +b_nx^n
$$

We define the product of these polynomials $c$ of degree $2n$ as:
$$
c(x)=c_0+c_1x+c_2x^2 \dots c_{2n}x^{2n}
$$

Our goal is to compute the coefficients $c_i$ for each $i$ from $0$ to $2n$. Naively using the distributive law:
$$
	c_i=\sum_{j=0}^{i}a_j \cdot b_{i-j}
$$
Thus, the computation takes $O(n)$ time per coefficient, yielding a net time complexity of $O(n^2)$ for all $2n$ coefficients. 

Nonetheless, we can parallelize this naive solution using CREW PRAM with $n$ processors yielding an effective $O(n)$ complexity. However, we will implement a better sequential algorithm and parallelize it further.

###### Coefficient and Point Representations
There are two ways an $n$ degree polynomial can be represented 
- with its $n + 1$ coefficients $-$ coefficient representation
- with $n + 1$ unique points which lie on its curve $-$ point representation 

In this project, we make use of the point representation. Consider the evaluation of the functions $a(x)$ and $b(x)$ at the points $[x_0, x_1, \dots x_{2n}]$:

$$
y_a := [a(x_0),a(x_1),a(x_2) \dots a(x_{2n})]
$$
$$
y_b := [b(x_0),b(x_1),b(x_2) \dots b(x_{2n})]
$$

One can easily find the function values $c(x_i)$ for each $i$ as
$$
c(x_i)=a(x_i) \cdot b(x_i) \;\;\;\; \text{or} \;\;\;\; c(x_i)=y_a[i]\cdot y_b[i]
$$
Once the arrays $y_a$ and $y_b$ are known, the computation of the array $$y_c := [c(x_0),c(x_1),c(x_2) \dots c(x_{2n})]$$can be done in $O(n)$ as each element is calculated in constant time.


##### A Divide-and-Conquer Approach: Fast Fourier Transform
The **Fast Fourier Transform (FFT)** is a sequential algorithm that allows the conversion of a general polynomial $P(x)$ from coefficient representation to point representation and vice versa efficiently.

Consider the evaluation of a polynomial $P(x)$ at $k$ points in the form $$y_P:=[P(x_1),P(x_2),P(x_3) \dots P(x_k)]$$Naively evaluating the array $y_P$ will take $O(n)$ time per point, thus $O(nk)$ in total.


To optimize the evaluation of point values, the sequential Fast Fourier Transform (FFT) algorithm is used.

We can decompose the polynomial into its **odd and even coefficients** as follows:
$$P(x) = P_{e}(x^2) + xP_{o}(x^2)$$
Where 
$$
\begin{align}
P_e(x) &= a_0 x^0 + a_2 x^1 + \dots + a_{n-2} x^{\frac{n}{2}-1} \\
P_o(x) &= a_1 x^0 + a_3 x^1 + \dots + a_{n-1} x^{\frac{n}{2}-1}
\end{align}
$$
We can say
$$P(-x) = P_{e}(x^2) - xP_{o}(x^2)$$
As we can see, the calculations of $P(x)$ and $P(-x)$ are nearly similar, but with slight variation. 
An example is shown below:
$$
P(x) = 3x^5 + 2x^4 + x^3 + 7x^2 + 5x + 1
$$

Evaluate at $n$ points $\pm x_1, \pm x_2, \dots, \pm x_{n/2}$

$$
P(x) = \underbrace{(2x^4 + 7x^2 + 1)}_{P_e(x^2)} + x \underbrace{(3x^4 + x^2 + 5)}_{P_o(x^2)}
$$

Thus, if we compute $P_e$ and $P_o$ at $n/2$ points, we can evaluate the polynomial at a positive-negative pair: $P(x)$ and $P(-x)$ for $n/2$ pairs of $x$, and thus get the values at $n$ points. Additionally, this calculation requires solving the same problem for $n/2$ points and for a polynomial of degree $n/2$ $-$ creating a divide and conquer recurrence. 

To calculate using such a partition of odd and even coefficients, we need positive-negative pairs throughout the recursion. As real numbers help us only up to the first recursion step, we make use of **complex numbers** such that throughout the recursion, among the points in consideration, for each $z$ we also have $-z=z \cdot e^{i \pi}$. Precisely, the function will be evaluated at the complex numbers: $$W_n:=[w_0,w_1,w_2 \dots w_{n}]$$where $w_i^{n}=1$ for all $i$ ($n^{th}$ roots of unity).


Consider $\omega = e^{2\pi i/n}$ 
$$
P(\omega^j) = P_e(\omega^{2j}) + \omega^j P_o(\omega^{2j})
$$
$$
P\left(\omega^{j + n/2}\right) = P_e(\omega^{2j}) - \omega^j P_o(\omega^{2j})
$$
$$
j \in \{0, 1, \dots, \frac{n}{2} - 1\}
$$

It is clear from the above equations that, the evaluation of $P_e$ and $P_o$ (recursively) helps us to find two function values namely $P(w^j)$ and $P(w^{j+n/2})$ simultaneously, as the two only differ by a '$-$' sign in between. 
Thus, we have obtained a divide and conquer recurrence. 
We can parallelize this divide and conquer algorithm with the help of threads. 

The recurrence will be as (using the **master theorem**): 
- $T(n) = 2 \;T(n/2) + O(n)$ for sequential $-$ which simplifies to $O(n\log{n})$.
- $T(n) = T(n/2) + O(1)$ for parallel $-$ which simplifies to $O(\log{n})$.

Thus, we have evaluated the polynomial at $n$ points. 

##### Conversion Back to Coefficient Representation: Inverse FFT

Consider the system for a polynomial $P$ of degree $n$ with coefficients given by the array $[p_i]$. Suppose that we have obtained the function values $P(w_j)$ where $w_j=\exp{(j \frac{2\pi i}{n})}$. Now, our goal is to obtain the array $[p_i]$ from the obtained values.

$\text{System 1:}$
$$
\begin{bmatrix}
    P(\omega^0) \\
    P(\omega^1) \\
    P(\omega^2) \\
    \vdots \\
    P(\omega^{n-1})
\end{bmatrix}
=
\begin{bmatrix}
    1 & 1 & 1 & \cdots & 1 \\
    1 & \omega & \omega^2 & \cdots & \omega^{n-1} \\
    1 & \omega^2 & \omega^4 & \cdots & \omega^{2(n-1)} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & \omega^{n-1} & \omega^{2(n-1)} & \cdots & \omega^{(n-1)(n-1)}
\end{bmatrix}
\begin{bmatrix}
    p_0 \\
    p_1 \\
    p_2 \\
    \vdots \\
    p_{n-1}
\end{bmatrix}
$$

$$
\text{or} \;\;\;\;
\begin{bmatrix}
    p_0 \\
    p_1 \\
    p_2 \\
    \vdots \\
    p_{n-1}
\end{bmatrix}
=
\begin{bmatrix}
    1 & 1 & 1 & \cdots & 1 \\
    1 & \omega & \omega^2 & \cdots & \omega^{n-1} \\
    1 & \omega^2 & \omega^4 & \cdots & \omega^{2(n-1)} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & \omega^{n-1} & \omega^{2(n-1)} & \cdots & \omega^{(n-1)(n-1)}
\end{bmatrix}^{-1}
\begin{bmatrix}
    P(\omega^0) \\
    P(\omega^1) \\
    P(\omega^2) \\
    \vdots \\
    P(\omega^{n-1})
\end{bmatrix}
$$

On simplifying the inverse, it can be written as:

$\text{System 2:}$
$$
\begin{bmatrix}
    p_0 \\
    p_1 \\
    p_2 \\
    \vdots \\
    p_{n-1}
\end{bmatrix}
=
\frac{1}{n}
\begin{bmatrix}
    1 & 1 & 1 & \cdots & 1 \\
    1 & \omega^{-1} & \omega^{-2} & \cdots & \omega^{-(n-1)} \\
    1 & \omega^{-2} & \omega^{-4} & \cdots & \omega^{-2(n-1)} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & \omega^{-(n-1)} & \omega^{-2(n-1)} & \cdots & \omega^{-(n-1)(n-1)}
\end{bmatrix}
\begin{bmatrix}
    P(\omega^0) \\
    P(\omega^1) \\
    P(\omega^2) \\
    \vdots \\
    P(\omega^{n-1})
\end{bmatrix}
$$

It turns out that, the computation of the coefficients $p_i$ here is very similar to the computation of the function values themselves (as presented in the $\text{System 1}$), except for the fact the $\omega$ has been replaced with $\frac{1}{n} \omega ^{-1}$. 

Conveniently, we can use the same algorithm that was used in computation of function values now to get the coefficients. With the above-mentioned changes, a version of the FFT is developed called **Inverse Fast Fourier Transform (IFFT)** with the same time complexity $O(n\log{n})$.

#### Use of FFT and Inverse FFT in Polynomial Multiplication

Given the above two algorithms, one can easily now perform polynomial multiplication by the following steps:

1. $\text{Compute}$ $f_a=\text{FFT}(a)$ and $f_b=\text{FFT}(b)$ $\text{as point representations}$ 
2. $\text{Calculate}$ $c'[i] = f_a[i] \cdot f_b[i] - \text{function values of}$ $c(x)$ $\text{at roots of unity}$
3. $\text{Compute}$ $c(x)$ $\text{in coefficient form by}$ $c[\dots]=\text{IFFT}(c')$ 

In the sequential implementation, step $1$ has a time complexity of $O(n\log{n})$ as discussed before, step $2$ has a complexity of $O(n)$ and the final step has a complexity of $O(n\log{n})$. This gives an overall time complexity of $O(n\log{n})$ for polynomial multiplication.

#### Implementation Details, Parallelization

##### Approach 1: Recursive Implementation
Firstly, we can distribute the iterations among the threads in step $2$ which leads to $O(1)$ time and $O(n)$ work. However, computation of FFT is also parallelizable. We can make use of the `section` clause in OpenMP to distribute the recursive calls between $2$ threads for even and odd parts. Then, to perform merging of results, consider the following snippet (in the parallel version):

```
void fft(a[...]) {
	... 
	
	#pragma omp parallel sections
	{
		#pragma omp section
		fft(a_even);
	
		#pragma omp section
		fft(a_odd);
	}
	
	#pragma omp parallel for
	for (int j = 0; j < n / 2; ++j) {
		w_j := e ^ {j * (2*PI*i/n)}
		a[j] = a_even[j] + w_j * a_odd[j];
		a[j + n / 2] = a_even[j] - w_j * a_odd[j];
	}
	
	...
}
```

We can see that the computations of the transform in the `for` loop can be independently done using a CREW PRAM using $n/2$ processors (or threads), leading to a time complexity of $O(1)$ with $O(n)$ work.

Effectively, FFT can be parallelized to yield a time complexity of $O(\log{n})$ (from the master theorem) and $O(n\log{n})$ work. 

Thus, the parallelized version of multiplication of polynomials is done in $O(\log{n})$ time and $O(n\log{n})$ work overall.

##### Approach 2: Iterative Implementation
We could see that the recursive implementation was performing satisfactorily, but nevertheless, we did experience some issues that were making it slower than expected during testing. The reasons were likely **thread management overhead** through the recursion and **stack usage**. We decided to improvise it, and then we arrived at an alternative implementation that is iterative in nature.

Consider the recurrence tree for a polynomial of degree $7:$

![[iterative-recurrence.svg]]

We can see that the ordering of the leaves (base cases of recursion) is in the order of 
$$
[rev_{0,k}, \;\; rev_{1,k}, \;\; rev_{2, k} \;\; \dots \;\; rev_{n-1,k}]
$$

Where $rev_{i,k}$ is the number obtained by **reversing the binary representation** string of the number $i$, when expressed up to $k$ digits. Here, $k=\log_2{n}=\log_2{8}=3$. Thus,

$$
\begin{equation}
	\begin{aligned}
		\text{indices} := [000, 001, 010, 011, 100, 101, 110, 111] \\
		\text{indices with reversal} := [000, 100, 010, 110, 001, 101, 011, 111]
	\end{aligned}
\end{equation}
$$

which gives the sequence $[0, 4, 2, 6, 1, 5, 3, 7$]. We need this sequence because this is the order of leaves in the recursion tree from left to right. We can use this order to implement the same algorithm iteratively.

So the iterative approach is as follows. We first obtain the sequence after reversal of indices, and permute the coefficients in that order. For example, the array 
$$
[a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_7]
$$
becomes
$$
[a_0,a_4,a_2,a_6,a_1,a_5,a_3,a_7]
$$
Then, we iterate in steps of sizes $\text{step} = 2^1, 2^2, 2^3 \dots \frac{n}{2} \;$ successively, and in each step, the the FFT value at positions $i$ and $i + \frac{\text{step}}{2}$ can be simultaneously computed by the positive-negative pair relationship described earlier. 

The code snippet below describes the exact implementation:

```
void fft(leaves[...]) {
    n := leaves.size();
    
    for (step = 2; step <= n; step *= 2) {
        angle := 2 * PI / step;
        wstep := Complex(cos(angle), sin(angle));

        for (int i = 0; i < n; i += step) {
            w := 1;

            for (int j = 0; j < step / 2; ++j) {
                Complex u = leaves[i + j];
                Complex v = leaves[i + j + step / 2] * w;
                leaves[i + j] = u + v;
                leaves[i + j + step / 2] = u - v;
	            
	            w = w * wstep    // increment angle; move to next point
            }
        }
    }
}
```

Effectively, the outer loop runs in $O(\log{n})$ time, the inner loops modify exactly $n$ values of the array, resulting in a time complexity of $O(n\log{n})$. 

When the inner loops are parallelized, they can be completed in $O(1)$ time with $n$ processors. This gives a time complexity of $O(\log{n})$ with $O(n\log{n})$ work.

### Performance Analysis

##### System Architecture
The below tests were performed on a computer with a 13th Gen Intel® Core™ i5-1340P processor with $16$ cores, and $16 \; \text{GB}$ of RAM. The shared memory model of PRAM is used here. The correct answers to check against were generated by the naive approach.

##### 1. Running Times

![[stats.png]]

Note: the y-axis is scaled in units of $\times 10^{-6}$ seconds (microsec.)

##### 2. Speed Up Analysis

![[speedup 1.png]]

As we can observe, the speed up of the parallel implementation turns out be as follows:

**Iterative versions:**
$$
speedup_{iter}=\frac{T_{\text{seq}}}{T(n,P)}=\frac{3038749 \; \mu s}{723337 \; \mu s}=420.101 \; \%
$$
**Recursive versions:**
$$
speedup_{rec}=\frac{T_{\text{seq}}}{T(n,P)}=\frac{4037975 \; \mu s}{2652270 \; \mu s}=152.246 \; \%
$$

when checked for the *largest* $n= 262144 \; (2^{18}$).
Overall, if checked for the plain sequential FFT (unoptimized, recursive) and fully optimized iterative parallel FFT algorithms, the speed up is:

$$
overall \; speedup=\frac{T_{\text{seq}}}{T(n,P)}=\frac{4037975 \; \mu s}{723337 \; \mu s}=558.243 \; \%
$$

###### Note: Correctness
Since we used the `long double` type in C++ for the best precision, for the test cases considered, the absolute error was less than $1 \; \%$. However, the use of the 86-bit floating point type does lead to a compromise in the running time of the all the programs.
### Conclusion

In this project, we presented the sequential recursive and iterative versions of the Fast Fourier Transform and its application with regard to multiplication of two polynomials. We also discussed the Inverse Fast Fourier Transform and its use in the conversion between the point and coefficient representations. We were also able to parallelize both versions in OpenMP and the results were analyzed resulting in a sufficiently acceptable correctness and performance.

Overall, the project showcased the effectiveness of leveraging parallel processing to accelerate the computation of the FFT and polynomial multiplication. The results provide a solid foundation for further optimizing and applying these techniques in real-world applications, such as real-time signal processing and communications.

### Future Work

- Exploring alternative parallelization techniques beyond OpenMP, such as GPU acceleration
- Identifying potential avenues for further optimizing the parallel algorithms for even greater speedups
- Integrating the parallel FFT into larger application frameworks or computational pipelines