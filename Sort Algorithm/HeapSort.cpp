//
// Created by Bruce Yang on 2023/9/17.
//

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void heapify(vector<int>& nums, int i, int n)
{
    int largest = i;
    int leftChild = 2*i + 1;
    int rightChild = 2*i + 2;

    if(leftChild < n && nums[leftChild] < nums[largest])
    {
        largest = leftChild;
    }
    if(rightChild < n && nums[rightChild] < nums[largest])
    {
        largest = rightChild;
    }

    if(largest != i)
    {
        swap(nums[largest], nums[i]);
        heapify(nums, largest, n);
    }
}

void heapSort(vector<int>& nums)
{
    int n = nums.size();

    for(int i=n/2-1; i>=0; --i)
    {
        heapify(nums, i, n);
    }

    for(int i=n-1; i>=0; --i)
    {
        swap(nums[0], nums[i]);
        heapify(nums, 0, n);
    }

}


int main()
{
    vector<int> nums = {12, 11, 13, 5, 6, 7};
    heapSort(nums);

    for(auto num: nums)
        cout << num << " ";
    cout << endl;
}
