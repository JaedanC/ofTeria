# 🌳 ofTeria 🌳

![ofTeria screenshot](https://github.com/jaedanc/ofTeria/raw/master/img/screenshot.png "ofTeria")

ofTeria is an [Openframeworks](https://openframeworks.cc/about/) based game designed around the game [Terraria](https://store.steampowered.com/app/105600/Terraria/) by [Relogic](https://re-logic.com/). By no means is it a replacement for the game, I just really enjoy the game. This my fourth attempt at creating this style of game and my first large C++ project so go easy on me.

If you'd like to give the game a try just clone this repo and open the `ofTeria.sln` file Visual Studio and compile in debug or release.

## 🎮 Controls 🎮

![settings.ini file](https://github.com/jaedanc/ofTeria/raw/master/img/settings_file.png "setting.ini file")

So far only basic controls are supported. You can move the player using the **arrow keys** or by **wasd**. The camera supports zooming which is through the **-** and **+** keys. These bindings can be remapped inside the `settings.ini` file located in `bin/settings.ini`. First declare the alias in the `[alias]` section and then define it to a key in the `[bindings]` section. If you wish to use a special key (eg. mouse1, rightArrow, tab etc) check the `KeyboardInput.cpp` for a list of preset defined special keys.

## ⬛ Console ⬛

![Console screenshot](https://github.com/jaedanc/ofTeria/raw/master/img/console_screenshot.png "Console screenshot")

ofTeria has a console that can be opened and closed using **~** (tilde). So far it supports two commands.

- `bind <key> <alias>`
- `clear`

Alias's that are supported are outlined in the `settings.ini` file. Try typing `bind f right` and then press f. `Ctrl + L` also clears the console. You can also select text by holding down shift. The text editor is designed to behave like typical designs like in Notepad etc.

## 🔄 GameStates 🔄

ofTeria works by using a [Finite State Machine](https://en.wikipedia.org/wiki/Finite-state_machine) stack. The `ofxGameEngine` stores a stack of GameStates and this can be pushed and popped from. `GameStates` can define whether they want States lower on the Stack to recieve `update()` or `draw()` calls from the Engine. For eg, a pause screen would just be a state that allows `draw()` calls but not `update()` calls through. This feature internally is called `updateTransparency` and `drawTransparency`.

## 🔌 Addons 🔌

ofTeria uses custom addons that I have created and some that are from other github projects.

### 📂 ofxMemoryMapping 📂

![ofxMemoryMapping addon](https://github.com/jaedanc/ofTeria/raw/master/img/ofxmemorymapping.png "ofxMemoryMapping addon")

This addon uses the Microsoft Api to simulate [mmap](https://en.wikipedia.org/wiki/Memory-mapped_file) from linux. `mmap` increases file io performance by 200x in some cases. This addon however only works for windows. It exposes familiar to use functions like `write`, `read` and the memory mapping is closed upon losing scope. `ofxMemoryMapping` also will resize the file mapping accordingly if you write past the end of the file removing the limitation of mmap where you need to specify the exact size of the file before you map the file. This addon is my most reused addon for any project that requires fast io performance.

### 🐛 ofxDebugger 🐛

![ofxDebugger addon](https://github.com/jaedanc/ofTeria/raw/master/img/ofxdebugger.png "ofxDebugger addon")

This addon exposes a macro that lets you push debug strings to the screen. It handles the drawing when you call `debugDraw()`. Push you debug strings to it by writing `debugPush(your_string)`

### 🧾 ofxIni 🧾

See [ofxIni on github](https://github.com/diederickh/ofxIni) by diederickh. Used for reading and writing to `.ini` files. Modified slightly so that it would compile haha 😂.

### 🐍 pystring 🐍

See [pystring on github](https://github.com/imageworks/pystring) by imageworks. Used for text manipulation in the in-game console.

### ⌛ ofxTimer ⌛

![ofxTimer addon](https://github.com/jaedanc/ofTeria/raw/master/img/ofxtimer.png "ofxTimer addon")

A Timer class that outputs the object's lifetime to `debugPush()` (See ofxDebugger). Create a timer by calling `ofxTimer your_timer("Timer label")`. This timer class is based on the Timer written by [The Cherno](https://www.youtube.com/user/TheChernoProject) on youtube in [this](https://www.youtube.com/watch?v=oEx5vGNFrLk) video.