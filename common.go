package ftrl

import (
	"math"
)

// sgn 符号函数
func sgn(val float64) float64 {
	if val > 0 {
		return 1
	} else if val < 0 {
		return -1
	}

	return 0
}

// sigmaFunc σ(a) = 1/(1 + exp(−a))
func sigmaFunc(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}
