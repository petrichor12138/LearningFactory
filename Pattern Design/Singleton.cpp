//
// Created by Bruce Yang on 2023/9/17.
//

// 饱汉式单例模式
class Singleton
{
public:
    static Singleton& getInstance()
    {
        // 静态局部变量，保证只初始化一次
        static Singleton instance;
        return instance;
    }

    // 拷贝构造和赋值被禁用
    Singleton(const Singleton&) = delete;
    Singleton& operator = (const Singleton&) = delete;

private:
    // 私有化构造器，禁止外部构造
    Singleton() {}
};

