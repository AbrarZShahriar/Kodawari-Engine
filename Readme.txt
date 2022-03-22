For Visual Studio:-
make build dir
go to build dir 
enter cmake -S ../ -B .

copy /third_party/SDL2-devel-2.0.20-VC/SDL2-2.0.20/lib/x64/SDL2.dll copy and paste beside the .exe after compilation.


Based on https://vkguide.dev/ vulkan tutorial. Did not add submodules as vkguide already had many things vendored in.

No textures as I haven't finished that part of the tutorial yet.


Current only has 1 tower. 

Victory condition- Score goes over 2000. Lose condition- Not yet inplemented.

WASD moves the tower placement cursor.

I tried to do something like "megastructs" by Ryan F(https://twitter.com/ryanjfleury) -https://www.youtube.com/watch?v=UolgW-Ff4bA
and HMH's sparse entity system - https://www.youtube.com/watch?v=wqpxe-s9xyw but haven't fully grokked it yet.




Todo-
Homing projectiles. 
Complete Tower placement grid.
Textures.
Particles.
Threadpools https://github.com/bshoshany/thread-pool




