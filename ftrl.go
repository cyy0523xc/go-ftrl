package ftrl

import (
	"math"
)

// 是否为测试模式
// 测试模式下：训练模型时会记录相关的测试信息
var debug bool
var debugResults []*PredictResult

// 预测结果
// 用于模型评估
type PredictResult struct {
	// 真实的观测值
	RealY float64

	// 预测的观测值
	PredictY float64
}

type FTRL struct {
	// 输入的初始化参数
	// alpha: Learning rate's proportionality constant.
	// beta: Learning rate's parameter.
	alpha, beta float64

	// L1范数：使特征稀疏化。值越大，越稀疏
	// L2范数：避免过拟合。值越大，泛化越好，越稀疏
	l1, l2 float64

	// 特征总数
	maxFeature int
}

// 一个观察实例
type Instance struct {
	// 非0特征数组
	Features []*Feature

	// 结果变量
	Y float64
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

// Input: parameters α, β, λ1 , λ2
// alpha:
// beta:
// l1:
// l2:
func New(alpha, beta, l1, l2 float64) *FTRL {
	return &FTRL{
		alpha: alpha,
		beta:  beta,
		l1:    l1,
		l2:    l2,
	}
}

// 修改测试状态
func SetDebug(isDebug bool) {
	debug = isDebug
}

func GetDebugResults() []*PredictResult {
	return debugResults
}

// Train 训练模型
// instances: 观测数组
// Receive feature vector xt and let features = {i | xi != 0}
func (f *FTRL) Train(maxFeature int, instances []*Instance) {
	// 初始化模型参数
	f.UpdateMaxFeature(maxFeature)

	var pt float64
	index := 0
	for _, instance := range instances {
		// 更新权重
		/*for i := 0; i < f.maxFeature; i++ {
			if math.Abs(z[i]) < f.l1 {
				w[i] = 0
			} else {
				w[i] = -(z[i] - sgn(z[i])*f.l1) / ((f.beta - math.Sqrt(n[i])/f.alpha) + f.l2)
			}
		}*/

		for _, feature := range instance.Features {
			i := feature.Index
			if math.Abs(z[i]) <= f.l1 {
				w[i] = 0
			} else {
				// 非0特征
				w[i] = -(z[i] - sgn(z[i])*f.l1) / ((f.beta - math.Sqrt(n[i])/f.alpha) + f.l2)
			}
		}

		// 预测
		pt = f.Predict(instance.Features)

		// 更新模型参数
		f.updateParams(instance.Features, pt, instance.Y)

		if debug {
			// 记录测试数据
			result := &PredictResult{
				RealY:    instance.Y,
				PredictY: pt,
			}
			debugResults = append(debugResults, result)

			index += 1
		}
	}
}

// UpdateMaxFeature 初始化最大特征数
func (f *FTRL) UpdateMaxFeature(maxFeature int) {
	if maxFeature > f.maxFeature {
		// 初始化z和n数组
		// (∀ i ∈ {1, . . . , d}), initialize zi = 0 and ni = 0
		for i := 0; i < maxFeature-f.maxFeature; i++ {
			z = append(z, 0)
			n = append(n, 0)
			w = append(w, 0)
		}

		f.maxFeature = maxFeature
	}
}

// Test 测试模型
func (f *FTRL) Test(instances []*Instance) []*PredictResult {
	var p float64
	for _, instance := range instances {
		// 预测
		p = f.Predict(instance.Features)

		result := &PredictResult{
			RealY:    instance.Y,
			PredictY: p,
		}
		debugResults = append(debugResults, result)
	}

	return debugResults
}

// Save 存储模型
func (f *FTRL) Save() {

}

// Load 加载模型
func (f *FTRL) Load() {

}

// Predict 预测
// features: 非0特征
// p_t = σ(x_t · w)
// σ(a) = 1/(1 + exp(−a))
func (f *FTRL) Predict(features []*Feature) float64 {
	var val float64
	for _, ft := range features {
		val += ft.Val * w[ft.Index]
	}

	return sigmaFunc(val)
}

// ***************** Private ******************************

// updateParams 更新模型参数
// features: 观测的特征数组
// pt: 预测的值
// yt: 实际值
func (f *FTRL) updateParams(features []*Feature, pt, yt float64) {
	var g, s, gSquare, xi float64
	var i int

	for _, ft := range features {
		i = ft.Index
		xi = ft.Val

		// g_i = (p_t − y_t ) * x_i  #gradient of loss w.r.t. w_i
		g = (pt - yt) * xi
		gSquare = g * g

		// σ_i = 1/α * (sqrt(n_i + g_i^2) − sqrt(n_i))  #equals 1/η_t,i − 1/η_t−1,i
		s = (math.Sqrt(n[i]+gSquare) - math.Sqrt(n[i])) / f.alpha

		// z_i ← z_i + g_i − σ_i * w_t,i
		z[i] += g - s*w[i]

		// n_i ← n_i + g_i ^ 2
		n[i] += gSquare
	}
}
