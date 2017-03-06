package ftrl

import (
	"fmt"
	"testing"
)

func TestTrain(t *testing.T) {
	var alpha, beta, l1, l2 float64
	alpha = 0.1
	beta = 4000
	l1 = 1
	l2 = 1

	SetDebug(true)
	f := New(alpha, beta, l1, l2)
	f.Train(trainMaxFeature, trainData)

	results := GetDebugResults()
	var totalY, totalN int
	var sumY, sumN float64
	for i := 0; i < len(results); i++ {
		results[i].PredictY = 10000 * results[i].PredictY
		if results[i].RealY > 0.5 {
			// 点击
			sumY += results[i].PredictY
			totalY += 1
		} else {
			sumN += results[i].PredictY
			totalN += 1
		}

		if i < 300150 && i > 300000 || i < 40 {
			fmt.Printf("Y: %d, P: %d\n", int(results[i].RealY), int(results[i].PredictY))
			if i == 300001 {
				// 输出权重
				fmt.Println("wwwwwwwwwwwwww")
				fmt.Println(w)
				fmt.Println("nnnnnnnnnnnnnn")
				fmt.Println(n)
				fmt.Println("zzzzzzzzzzzzzz")
				fmt.Println(z)
			}
		}
	}

	fmt.Printf("totalY: %d, avgY: %f\n", totalY, sumY/float64(totalY))
	fmt.Printf("totalN: %d, avgN: %f\n", totalN, sumN/float64(totalN))
	fmt.Printf("Rate: %f\n", float64(totalY)/float64(totalY+totalN))
}
