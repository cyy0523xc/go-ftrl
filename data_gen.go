package ftrl

type DataGen struct {
	// 特征数量
	max_feature int

	// 特征文件路径
	path string
}

func NewDataGen(path string) *DataGen {
	return &DataGen{
		path: path,
	}
}

// 解释特征文件
func (d *DataGen) Parse() {

}

// 获取一个观测
func (d *DataGen) GetRow() ([]Feature, int) {
	return nil
}
