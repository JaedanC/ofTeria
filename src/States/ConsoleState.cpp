#include "ConsoleState.h"
#include "ofxDebugger/ofxDebugger.h"
#include "../Keyboard/KeyboardInput.h"

ConsoleState ConsoleState::instance;

void ConsoleState::keyPressed(int key)
{
	/*
	~man ascii
		30 40 50 60 70 80 90 100 110 120
		---------------------------------
	0:    (  2  <  F  P  Z  d   n   x
	1:    )  3  =  G  Q  [  e   o   y
	2:    *  4  >  H  R  \  f   p   z
	3: !  +  5  ?  I  S  ]  g   q   {
	4: "  ,  6  @  J  T  ^  h   r   |
	5: #  -  7  A  K  U  _  i   s   }
	6: $  .  8  B  L  V  `  j   t   ~
	7: %  /  9  C  M  W  a  k   u  DEL
	8: &  0  :  D  N  X  b  l   v
	9: '  1  ;  E  O  Y  c  m   w
	*/
	if (key >= ' ' && key <= '~') {
		currentCommand += key;
	}

	/*
	Controlling the cursor. Enter, Backspace and arrow keys.
	*/
	switch (key) {
	case OF_KEY_RETURN:
		submitCommand(currentCommand);
		currentCommand.clear();
		break;
	case OF_KEY_BACKSPACE:
		if (currentCommand.size() > 0) {
			currentCommand.pop_back();
		}
		break;
	case OF_KEY_UP:
		lastHistoryMarker = ofWrap(--lastHistoryMarker, -1, history.size());
		currentCommand = (lastHistoryMarker == -1) ? "" : history[lastHistoryMarker];
		break;
	case OF_KEY_DOWN:
		lastHistoryMarker = ofWrap(++lastHistoryMarker, -1, history.size());
		currentCommand = (lastHistoryMarker == -1) ? "" : history[lastHistoryMarker];
		break;
	}

	// TODO remove
	cout << "Key from callback pressed " << key << endl;
}

void ConsoleState::keyReleased(int key)
{
	// TODO remove
	cout << "Key from callback released " << key << endl;
}

void ConsoleState::setup()
{
	KeyboardInput::Instance()->registerPressedCallback(this);
	screenPos = { ofGetWidth() - 300.0f, 15.0f };
	width = 285;
	height = 200;

	// TODO: Remove these three lines for testing
	history.push_back("Hello world");
	history.push_back("Second command");
	registerAliasBlock("jump", INPUT_BLOCK);
}

void ConsoleState::update(ofxGameEngine* game)
{
	// TODO: Remove these checks for Alias'
	/*
	if (KeyboardInput::Instance()->queryAliasPressed("jump")) {
		cout << "Alias for jump pressed!\n";
	}
	if (KeyboardInput::Instance()->queryAliasDown("jump")) {
		cout << "Alias for jump down!\n";
	}
	if (KeyboardInput::Instance()->queryAliasReleased("jump")) {
		cout << "Alias for jump released!\n";
	}
	*/
	if (queryAliasPressed(string("jump"))) {
		cout << "New query triggered for jump\n";
	}
}

void ConsoleState::draw(ofxGameEngine* game)
{
	debugPush("State: ConsoleState");
	debugPush("HistoryMarker: " + ofToString(lastHistoryMarker));


	// Draw the console history
	ofSetColor(ofColor::black);
	ofDrawRectangle(screenPos, width, height);
	ofSetColor(ofColor::white);
	ofDrawBitmapString(currentCommand, screenPos.x + 5, screenPos.y + 15 * 1);
	for (unsigned int i = 0; i < history.size(); i++) {
		string& command = history[history.size() - 1 - i];
		ofDrawBitmapString(command, screenPos.x + 5, screenPos.y + 15 * (i + 2));
	}

	// Draw a typing marker
	// (Width : 8pt , Height : 11pt ) each character.
	ofSetLineWidth(1.0f);
	ofDrawLine(
		screenPos + ofVec2f{currentCommand.size() * 8 + 7.0f, 5.0f},
		screenPos + ofVec2f{currentCommand.size() * 8 + 7.0f, 11 + 5.0f}
	);
	//ofDrawLine(screenPos.x + currentCommand.size() * 8 + 7, screenPos.y + 5, screenPos.x + currentCommand.size() * 8 + 7, screenPos.y + 11 + 5);
}

void ConsoleState::submitCommand(string& command)
{
	history.push_back(command);
	cullHistory(maxHistorySize);
}

/*
Clears the history queue
*/
void ConsoleState::clearHistory()
{
	for (unsigned int i = 0; i < history.size(); i++) {
		history.pop_front();
	}
}

/*
Limits the size of the history queue to be 'limit' long
*/
void ConsoleState::cullHistory(unsigned int limit)
{
	for (unsigned int i = limit; i < history.size(); i++) {
		history.pop_front();
	}
}

