import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 数据（样本）
x_data = np.array([0.1,0.3,0.5,0.7,0.9])  
y_data = np.array([-0.0333,0.0506,0.2633,0.4322,0.7398])          

# 构造线性模型，我们的目的是让 TensorFlow 帮我们求得 w 与 b 的具体值
w = tf.Variable(0.0) # 权重变量
b = tf.Variable(0.0) # 偏置值变量
y = w * x_data + b

# 损失函数
#
# tf.square(y_data - y) 求平方差 
# reduce_mean           求平均值 
loss = tf.reduce_mean(tf.square(y_data - y))

# 优化器 -- 梯度下降法
optimizer =  tf.train.GradientDescentOptimizer(0.1)

# 训练使loss"最小",即 使预测出来的 w 和 b "最优"
train = optimizer.minimize(loss)

# 常用写法（无需手动关闭会话）
with tf.Session() as sess:
    # 为了让过程更加形象，这里将计算过程绘制出来（plt用法在这不做介绍）
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # 绘制样本点
    ax.scatter(x_data,y_data)
    plt.ion()
    plt.show()  # 绘制完即阻塞
    
     # 初始化全部变量
    sess.run(tf.global_variables_initializer())
    
    # 训练1000次
    for step in range(1001):
        sess.run(train)
        if(step % 50 == 0):
            result = sess.run([w,b])  # 预测结果
            print(result)
            
            # 画线 （绘图相关，可不关注）
            try:
                ax.lines.remove(lines[0])   # 移除（历史）第一条线
            except Exception:
                pass
            # 预测结果用红色实线绘制       
            lines = ax.plot(x_data, result[0] * x_data + result[1], 'r-', lw=2)
            plt.pause(0.5)