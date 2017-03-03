package ftrl

import (
	"math"
)

type FTRL struct {
	// 输入的初始化参数
	// alpha: Learning rate's proportionality constant.
	// beta: Learning rate's parameter.
	alpha, beta float64

	// L1: regularization constant
	// L2: regularization constant
	l1, l2 float64

	// alpha参数的倒数
	alphaReciprocal float64

	// 特征总数
	max_feature int
}

// 一个特征
// 注：非0的特征才需要保存
type Feature struct {
	Index int     // 特征的下标
	Val   float64 // 特征值
}

// 特征权重
var w []float64

// 权重参数
// Lists and dictionaries to hold the weights. Initiate
// the weight vector z and learning rate n as None so that
// when self.train is called multiple times it will not
// overwrite the stored values. This essentially allows epoch
// training to take place, albeit a little bit ugly.
var z, n []float64

// 运算过程需要的参数
var g, sigma []float64

// Input: parameters α, β, λ1 , λ2
// alpha:
// beta:
// lambda1:
// lambda2:
func New(alpha, beta, lambda1, lambda2 float64) *FTRL {
	return &FTRL{
		alpha: alpha,
		beta:  beta,
		l1:    lambda1,
		l2:    lambda2,

		// alpha参数的倒数
		alphaReciprocal: 1 / alpha,
	}
}

// 训练模型
func (f *FTRL) Train(dataGen *DataGen) {
	// 初始化模型参数
	f.initMaxFeature(dataGen.max_feature)

	// features: 不为0的特征数组
	// Receive feature vector xt and let features = {i | xi != 0}
	for features, yt := range dataGen.GetRow() {
		f.Update(features, yt)
	}
}

// Update 更新模型
// features: 观测的特征数组
// yt: 实际值
func (f *FTRL) Update(features []Feature, yt float64) {
	// 更新权重
	for i := 0; i < f.max_feature; i++ {
		if math.Abs(z[i]) <= f.l1 {
			w[i] = 0
		} else {
			w[i] = -1 / (f.beta - math.Sqrt(n[i])/f.alpha + f.l2) * (z[i] - math.Signbit(z[i])*l1)
		}
	}

	// 预测
	// p_t = σ(x_t · w)
	// σ(a) = 1/(1 + exp(−a))
	pt = sigmaFunc(listMult(features, w))

	// 更新模型参数
	f.updateParams(features, pt, yt)
}

// updateParams 更新模型参数
// features: 观测的特征数组
// pt: 预测的值
// yt: 实际值
func (f *FTRL) updateParams(features []Feature, pt, yt float64) {
	for ft := range features {
		i := ft.Index
		xi := ft.Val

		// g_i = (p_t − y_t ) * x_i  #gradient of loss w.r.t. w_i
		g[i] = (pt - yt) * xi
		giSquare = g[i] * g[i]

		// σ_i = 1/α * (sqrt(n_i + g_i^2) − sqrt(n_i))  #equals 1/η_t,i − 1/η_t−1,i
		sigma[i] = f.alphaReciprocal * (math.Sqrt(n[i]+giSquare) - math.Sqrt(n[i]))

		// z_i ← z_i + g_i − σ_i * w_t,i
		z[i] = z[i] + g[i] - sigma[i]*w[i]

		// n_i ← n_i + g_i ^ 2
		n[i] = n[i] + giSquare
	}
}

// 更新模型
func (f *FTRL) Test() {

}

// 更新模型
func (f *FTRL) Predict(features []Feature) {
	var val float64 = 0
	for ft := range features {
		val += ft.Val * w[ft.Index]
	}

	return sigmaFunc(val)
}

// 初始化最大特征数
func (f *FTRL) initMaxFeature(max_feature) {
	if max_feature > f.max_feature {
		// 初始化z和n数组
		// (∀ i ∈ {1, . . . , d}), initialize zi = 0 and ni = 0
		for i := 0; i < max_feature-f.max_feature; i++ {
			z = append(z, 0)
			n = append(n, 0)
			g = append(g, 0)
			w = append(w, 0)
			sigma = append(sigma, 0)
		}

		f.max_feature = max_feature
	}
}

// σ(a) = 1/(1 + exp(−a))
func sigmaFunc(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}
