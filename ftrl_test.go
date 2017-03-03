package ftrl

import (
	"fmt"
	"testing"
)

func TestTrain(t *testing.T) {
	var alpha, beta, lambda1, lambda2 float64
	alpha = 0.5
	beta = 0.5
	lambda1 = 0.1
	lambda2 = 0.5

	SetDebug(true)
	f := New(alpha, beta, lambda1, lambda2)
	f.Train(trainMaxFeature, trainData)

	results := GetDebugResults()
	for i := 0; i < len(results); i++ {
		results[i].PredictY = 1000 * results[i].PredictY

		if i < 300100 && i > 300000 {
			fmt.Printf("Y: %d, P: %d\n", int(results[i].RealY), int(results[i].PredictY))
		}
	}

}
