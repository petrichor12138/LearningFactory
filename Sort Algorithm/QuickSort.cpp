//
// Created by Bruce Yang on 2023/9/17.
//
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void quickSort(vector<int>& nums, int l, int r)
{
    if(l >= r)
    {
        return ;
    }

    int i = l;
    int j = r;
    int pivot = nums[l];

    while(i < j)
    {
        while(i < j && nums[j] >= pivot)
            -- j;
        nums[i++] = nums[j];
        while(i < j && nums[i] <= pivot)
            ++ i;
        nums[j--] = nums[i];
    }

    nums[i] = pivot;
    quickSort(nums, l, i-1);
    quickSort(nums, i+1, r);
}

int main()
{
    vector<int> nums = {12, 11, 13, 5, 6, 7};
    quickSort(nums, 0, nums.size()-1);

    for(auto num: nums)
        cout << num << " ";
    cout << endl;
}