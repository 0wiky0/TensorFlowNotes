import tensorflow as tf
# 创建常量 op
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
# 创建一个矩阵乘法op， 把m1和m2传入
product = tf.matmul(m1,m2)
print(product)

# # 定义一个会话，启动默认的图
# sess = tf.Session()
# # 执行以上的op
# result = sess.run(product)
# print(result)
# sess.close()


# 常用写法（无需手动关闭会话）
with tf.Session() as sess:
    # 执行以上的op
    result = sess.run(product)
    print(result)