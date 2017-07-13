https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html

# python, numpy で使用する　axis
pythonのnumpy　で行列を連結するときに、知っておかないといけない　オプションの引数axis。

axisで、どの軸（行方向、列方向）に連結させるかを決めることができる。
デフォルトでは、axix = 0

arr = np.arange(12)
arr = arr.reshape((3,4))
	array([[0,1,2,3],
		[4,5,6,7],
		[8,9,10,11]])

np.concatenate([arr,arr],axis=0)

	array([[0,1,2,3],
 		[4,5,6,7],
 		[8,9,10,11,12],
 		[0,1,2,3],
 		[4,5,6,7],
 		[8,9,10,11]])

np.concatenate([arr,arr],axis=1)

	array([[0,1,2,3,0,1,2,3],
		[4,5,6,7,4,5,6,7],
		[8,9,10,11,8,9,10,11]])
