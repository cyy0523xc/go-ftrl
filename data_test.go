package ftrl

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	//"testing"
)

// 源数据
var srcFilename = "./libsvm_data.txt"

// 训练数据
var trainData []*Instance

// 测试数据
var testData []*Instance

var trainMaxFeature int

func init() {
	// 读入数据文件，并分成测试集和训练集
	file, err := os.Open(srcFilename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	rd := bufio.NewReader(file)
	var i int
	for {
		line, err := rd.ReadString('\n') //以'\n'为结束符读入一行
		if err != nil || io.EOF == err {
			break
		}

		instance, max := parseLibsvmDataLine(line)
		if max > trainMaxFeature {
			trainMaxFeature = max
		}

		if i&3 != 3 {
			trainData = append(trainData, instance)
		} else {
			testData = append(testData, instance)
		}

		i += 1
	}

	// 输出
	fmt.Printf("trainData: %d, testData: %d, maxFeature: %d\n", len(trainData), len(testData), trainMaxFeature)
	for _, i := range trainData[0].Features {
		fmt.Printf("feature index: %d, val: %f\n", i.Index, i.Val)
	}
	fmt.Println("Y: ", trainData[0].Y)
}

// ParseLibsvmDataLine 解释libsvm格式的数据
// line: 一行观察数据，格式如：1 16:1 19:1 24:1 26:1 58:1 682:1
// return:
// instance:
// maxIndex: 最大的特征下标
func parseLibsvmDataLine(line string) (instance *Instance, maxIndex int) {
	var arr []string
	arr = strings.Split(line, " ")
	instance = &Instance{
		Y:        float64(toInt(arr[0])),
		Features: make([]*Feature, 0),
	}

	var feature *Feature
	for i := 1; i < len(arr); i++ {
		ft := strings.Split(arr[i], ":")
		val, _ := strconv.ParseFloat(ft[1], 64)
		feature = &Feature{
			Index: toInt(ft[0]),
			Val:   val,
		}

		instance.Features = append(instance.Features, feature)

		if feature.Index+1 > maxIndex {
			maxIndex = feature.Index + 1
		}
	}

	return
}

func toInt(s string) int {
	i, _ := strconv.Atoi(s)
	return i
}
