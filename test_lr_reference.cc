#include <iostream>
#include <vector>

class A {
    public:
        A(std::vector<int> vec) : vec_(vec) {std::cout << "constructor"<<std::endl;}
        A(A& a) : vec_(a.vec_) {std::cout << "copy constructor"<<std::endl;}
        A(A&& a) : vec_(a.vec_) {std::cout << "move constructor"<<std::endl;}
        A& operator= (A& other) {
            std::cout << "copy assignment operator" << std::endl;
            vec_ = other.vec_;
            return *this;
        }
        A& operator= (A&& other) {
            std::cout << "move assignment operator" << std::endl;
            vec_ = other.vec_;
            return *this;
        }
        void print() {std::cout << vec_[0]<<std::endl;}
        void set() {vec_[0] = 999;}
    private:
        std::vector<int> vec_;
};

int main() {
    std::vector<int> vec(100,10);

    A a(vec);
    A b(a);
    A c = a;
    c = a;
    A d = std::move(a);
    d.print();
    d.set();
    d.print();
    // std::cout << a.vec_[10] << std::endl;
    d = std::move(a);
    d.print();
    return 0;
}
