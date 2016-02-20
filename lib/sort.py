# -*- coding: utf-8 -*-
#!/usr/bin/python
'''
Created on 2015-5-31
  
@author: csuldw
'''
  
def bubble_sort(seq):
    for i in range(len(seq)):
        for j in range(i,len(seq)):
            if seq[j] < seq[i]:
                tmp = seq[j]
                seq[j] = seq[i]
                seq[i] = tmp
                  
def selection_sort(seq):
    for i in range(len(seq)):
        position = i
        for j in range(i,len(seq)):
            if seq[position] > seq[j]:
                position = j
        if position != i:
                tmp = seq[position]
                seq[position] = seq[i]
                seq[i] = tmp
  
def insertion_sort(seq):
    if len(seq) > 1:
        for i in range(1,len(seq)):
            while i > 0 and seq[i] < seq[i-1]:
                tmp = seq[i]
                seq[i] = seq[i-1]
                seq[i-1] = tmp
                i = i - 1
                  
if __name__ == "__main__":
    print "--------bubble_sort-------------"
    seq = [21,3,33,41,7,6,8,9,11]
    bubble_sort(seq)
    print seq
    print "--------selection_sort-------------"
    seq = [88,44,33,5,15,6,8,9,11]
    selection_sort(seq)
    print seq
    print "--------insertion_sort-------------"
    seq = [777,44,33,54,17,26,1111,100,11]
    insertion_sort(seq)
    print seq