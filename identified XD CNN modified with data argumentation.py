import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10

# 加载 CIFAR-10 数据集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 数据归一化到 [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# 数据增强
# 优化的数据增强策略
datagen = ImageDataGenerator(
    rotation_range=15,             # 限制旋转角度，避免过度旋转
    width_shift_range=0.2,         # 水平平移
    height_shift_range=0.2,        # 垂直平移
    zoom_range=0.1,                # 稍微缩放
    horizontal_flip=True,          # 水平翻转
    shear_range=0.2,               # 错切变换
    brightness_range=[0.9, 1.1],   # 稍微调整亮度
    fill_mode='nearest'            # 填充模式，避免图像失真
)


# 适应训练数据
datagen.fit(X_train)

# 改进的CNN模型结构
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # 防止过拟合
    
    # 将全连接层的输出转换为序列，以便LSTM处理
    Reshape((-1, 512)),  # 将输出展平为一个序列
    LSTM(128, activation='relu', return_sequences=False),  # 添加LSTM层
    
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 学习率调度器
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# 训练模型
history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# 绘制损失函数和准确率曲线
plt.figure(figsize=(12, 6))

# 损失函数曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 评估模型并输出准确率
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
