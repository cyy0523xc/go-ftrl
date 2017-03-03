package ftrl

import (
	"math"
)

const valMax float64 = 35 // 28
const valMin float64 = -valMax
const one float64 = 1.0

// sgn 符号函数
func sgn(val float64) float64 {
	if val >= 0 {
		return 1
	}

	return 0
}

// sigmaFunc σ(a) = 1/(1 + exp(−a))
func sigmaFunc(val float64) float64 {

	if val > valMax {
		val = valMax
	} else if val < valMin {
		val = -valMin
	}

	//println(val)
	return one / (one + math.Exp(-val))
}
