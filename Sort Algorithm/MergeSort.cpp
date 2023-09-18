//
// Created by Bruce Yang on 2023/9/18.
//

#include <iostream>
#include <vector>

using namespace std;

void mergeSort(vector<int>& nums, int l, int r)
{
    if(l >= r)
        return ;

    int mid = l + (r - l)/2;
    mergeSort(nums, l, mid);
    mergeSort(nums, mid+1, r);

    vector<int> temp;
    int i = l;
    int j = mid + 1;

    while(i <= mid && j <= r)
    {
        if(nums[i] <= nums[j])
            temp.emplace_back(nums[i++]);
        else
            temp.emplace_back(nums[j++]);
    }

    while(i <= mid)
    {
        temp.emplace_back(nums[i++]);
    }
    while(j <= r)
    {
        temp.emplace_back(nums[j++]);
    }

    for(i=0; i<temp.size(); ++i)
    {
        nums[i+l] = temp[i];
    }

}

int main()
{
    vector<int> nums = {12, 11, 13, 5, 6, 7};
    mergeSort(nums, 0, nums.size()-1);

    for(auto num: nums)
        cout << num << " ";
    cout << endl;
}
