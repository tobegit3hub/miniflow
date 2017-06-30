/* Copyright 2017 The Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <time.h>
double My_variable = 3.0;
 
int fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);
}
 
int my_mod(int x, int y) {
    return (x%y);
}
 	
char *get_time()
{
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);
}
