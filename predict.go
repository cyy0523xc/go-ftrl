package ftrl

import (
	"sync"
)

type Predict struct {
	weight []float64 // 权重
	rw     sync.RWMutex
}

func NewPredict() *Predict {
	return &Predict{}
}

// UpdateWeight 更新权重
func (p *Predict) UpdateWeight(weight []float64) {
	p.rw.Lock()
	p.weight = weight
	p.rw.Unlock()
}

// Predict 预测
func (p *Predict) Predict(features []Feature) float64 {
	var val float64
	p.rw.RLock()
	for _, ft := range features {
		val += ft.Val * p.weight[ft.Index]
	}
	p.rw.RUnlock()

	return sigmaFunc(val)
}
