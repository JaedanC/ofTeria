#include "ConsoleState.h"
#include "ofxDebugger/ofxDebugger.h"
#include "../Keyboard/KeyboardInput.h"
#include "Console/ConsoleParser.h"
#include "../addons/pystring/pystring.h"

ConsoleState ConsoleState::instance;

ConsoleState::ConsoleState() : ofxGameState(
	false, // updateTransparent
	true, // drawTransparent
	"ConsoleState"
), width(285), height(200), consoleParser(this)
{
	// TODO: Remove these three lines for testing
	submitCommand(string("Hello world"));
	submitCommand(string("Second command"));
	registerAliasBlock("jump", INPUT_BLOCK);
}

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
	if (key >= ' ' && key <= '~' && key != '`') {
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
		commandHistoryMarker = ofWrap(--commandHistoryMarker, -1, commandHistory.size());
		currentCommand = (commandHistoryMarker == -1) ? "" : commandHistory[commandHistoryMarker];
		break;
	case OF_KEY_DOWN:
		commandHistoryMarker = ofWrap(++commandHistoryMarker, -1, commandHistory.size());
		currentCommand = (commandHistoryMarker == -1) ? "" : commandHistory[commandHistoryMarker];
		break;
	case OF_KEY_PAGE_UP:
		consoleHistoryMarker = ofClamp(--consoleHistoryMarker, 0, consoleHistory.size());
		break;
	case OF_KEY_PAGE_DOWN:
		consoleHistoryMarker = ofClamp(++consoleHistoryMarker, 0, consoleHistory.size());
		break;
	}
}

void ConsoleState::keyReleased(int key)
{
	// TODO remove
	cout << "Key from callback released " << key << endl;
}

void ConsoleState::setup()
{
	screenPos = { ofGetWidth() - 300.0f, 15.0f };
	ofxGameEngine::Instance()->getKeyboardInput()->registerKeyPressedCallback(this);
}

void ConsoleState::update(ofxGameEngine* game)
{
	// TODO: Remove
	if (queryAliasPressed("jump")) {
		cout << "New query triggered for jump\n";
	}

	if (queryDown(OF_KEY_CONTROL) && queryAliasPressed("jump")) {
		cout << "Control jump!\n";
	}

	if (queryAliasPressed("toggleConsole")) {
		cout << "Popping Console State\n";
		ofxGameEngine::Instance()->PopState();
	}
}

void ConsoleState::draw(ofxGameEngine* game)
{
	debugPush("State: ConsoleState");
	debugPush("HistoryMarker: " + ofToString(commandHistoryMarker));


	// Draw the console history
	ofSetColor(ofColor::black);
	ofDrawRectangle(screenPos, width, height);
	ofSetColor(ofColor::white);
	ofDrawBitmapString(currentCommand, screenPos.x + 5, screenPos.y + 15 * 1);

	int linesDrawn = 0;
	for (unsigned int i = consoleHistoryMarker; i < consoleHistory.size(); i++) {
		ConsoleEntry& entry = consoleHistory[consoleHistory.size() - 1 - i];

		ofSetColor(entry.colour);
		for (string& line : entry.message) {
			ofDrawBitmapString(line, screenPos.x + 5, screenPos.y + 15 * (linesDrawn + 2));

			linesDrawn++;
			if (linesDrawn >= showHistoryLines) {
				goto endloop;
			}
		}
	}
endloop:

	// Draw a typing marker
	// (Width : 8pt , Height : 11pt ) each character.
	ofSetLineWidth(1.0f);
	ofDrawLine(
		screenPos + ofVec2f{currentCommand.size() * 8 + 7.0f, 5.0f},
		screenPos + ofVec2f{currentCommand.size() * 8 + 7.0f, 11 + 5.0f}
	);
	//ofDrawLine(screenPos.x + currentCommand.size() * 8 + 7, screenPos.y + 5, screenPos.x + currentCommand.size() * 8 + 7, screenPos.y + 11 + 5);
}

void ConsoleState::exit()
{
	ofxGameEngine::Instance()->getKeyboardInput()->deregisterKeyPressedCallback(this);
}

void ConsoleState::submitCommand(string& command)
{
	vector<string> parameters;
	pystring::split(command, parameters);

	consoleParser.run(parameters);
	commandHistory.push_back(command);

	vector<string> commandVector = { command };
	consoleHistory.emplace_back(commandVector, ofColor(255, 255, 255));

	cullHistory();
}

void ConsoleState::addText(ConsoleEntry& entry)
{
	consoleHistory.push_back(entry);
}

void ConsoleState::addText(vector<string>& strings, ofColor colour)
{
	ConsoleEntry entry = { strings, colour };
	addText(entry);
}

void ConsoleState::addText(string& entry, ofColor colour)
{
	vector<string> entries = { entry };
	addText(entries, colour);
}

void ConsoleState::clearHistory()
{
	commandHistory.clear();
	consoleHistory.clear();
}

void ConsoleState::cullHistory()
{
	for (unsigned int i = commandHistoryMaxSize; i < commandHistory.size(); i++) {
		commandHistory.pop_front();
	}

	for (unsigned int i = consoleHistoryMaxSize; i < consoleHistory.size(); i++) {
		consoleHistory.pop_front();
	}
}

