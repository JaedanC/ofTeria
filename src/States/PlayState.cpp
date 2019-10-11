#include "PlayState.h"
#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ConsoleState.h"
#include "../Keyboard/KeyboardInput.h"

void PlayState::setup()
{
	ofxGameEngine::Instance()->getKeyboardInput()->registerAlias("shoot", "q");
}

void PlayState::update(ofxGameEngine* game)
{
	if (queryInput("jump")) {
		cout << "PlayState is not getting blocked!\n";
	}
	if (queryInput("toggleConsole")) {
		ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	}

	getWorldSpawn().lock()->update();
}

void PlayState::draw(ofxGameEngine* game)
{
	getWorldSpawn().lock()->draw();
}
