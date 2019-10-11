#include "KeyboardInput.h"

KeyboardInput::KeyboardInput()
{
	specialBindings.reserve(42);

	specialBindings["lmouse"] = OF_MOUSE_BUTTON_LEFT + MOUSE_OFFSET_CONSTANT;
	specialBindings["rmouse"] = OF_MOUSE_BUTTON_RIGHT + MOUSE_OFFSET_CONSTANT;
	specialBindings["mmouse"] = OF_MOUSE_BUTTON_MIDDLE + MOUSE_OFFSET_CONSTANT;

	specialBindings["mouse1"] = OF_MOUSE_BUTTON_1 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse2"] = OF_MOUSE_BUTTON_2 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse3"] = OF_MOUSE_BUTTON_3 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse4"] = OF_MOUSE_BUTTON_4 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse5"] = OF_MOUSE_BUTTON_5 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse6"] = OF_MOUSE_BUTTON_6 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse7"] = OF_MOUSE_BUTTON_7 + MOUSE_OFFSET_CONSTANT;
	specialBindings["mouse8"] = OF_MOUSE_BUTTON_8 + MOUSE_OFFSET_CONSTANT;

	specialBindings["space"] = ' ';
	specialBindings["' '"] = specialBindings["space"];
	specialBindings["ctrl"] = OF_KEY_CONTROL;
	specialBindings["shift"] = OF_KEY_SHIFT;
	specialBindings["alt"] = OF_KEY_ALT;
	specialBindings["tab"] = OF_KEY_TAB;
	specialBindings["return"] = OF_KEY_RETURN;
	specialBindings["enter"] = specialBindings["return"];

	specialBindings["esc"] = OF_KEY_ESC;
	specialBindings["home"] = OF_KEY_HOME;
	specialBindings["end"] = OF_KEY_END;
	specialBindings["pgup"] = OF_KEY_PAGE_UP;
	specialBindings["pgdown"] = OF_KEY_PAGE_DOWN;
	specialBindings["del"] = OF_KEY_DEL;
	specialBindings["insert"] = OF_KEY_INSERT;

	specialBindings["up"] = OF_KEY_UP;
	specialBindings["down"] = OF_KEY_DOWN;
	specialBindings["left"] = OF_KEY_LEFT;
	specialBindings["right"] = OF_KEY_RIGHT;

	specialBindings["f1"] = OF_KEY_F1;
	specialBindings["f2"] = OF_KEY_F2;
	specialBindings["f3"] = OF_KEY_F3;
	specialBindings["f4"] = OF_KEY_F4;
	specialBindings["f5"] = OF_KEY_F5;
	specialBindings["f6"] = OF_KEY_F6;
	specialBindings["f7"] = OF_KEY_F7;
	specialBindings["f8"] = OF_KEY_F8;
	specialBindings["f9"] = OF_KEY_F9;
	specialBindings["f10"] = OF_KEY_F10;
	specialBindings["f11"] = OF_KEY_F11;
	specialBindings["f12"] = OF_KEY_F12;
}

int KeyboardInput::convertStringToKey(const string& str) {
	if (str.empty()) {
		cout << "KeyboardInput::convertStringToKey() returning 0. str is empty\n";
		return 0;
	}

	if (specialBindings.count(str) == 0) {
		return str[0];
	}

	return specialBindings[str];
}

void KeyboardInput::keyPressed(int key)
{
	// Update keyPressed Maps
	keyDownMap[key] = true;
	keyPressedMap[key] = true;

	// Run callback functions for the instances that have registered to recieve them also
	for (auto& callback : keyPressedCallbacks) {
		callback->keyPressed(key);
	}
}

void KeyboardInput::keyReleased(int key)
{
	// Update keyPressed Maps
	keyDownMap[key] = false;
	keyReleasedMap[key] = true;

	// Run callback functions for the instances that have registered to recieve them also
	for (auto& callback : keyReleasedCallbacks) {
		callback->keyReleased(key);
	}
}

void KeyboardInput::mousePressed(int x, int y, int button)
{
	// Update mousePressed Maps
	keyDownMap[button + MOUSE_OFFSET_CONSTANT] = true;
	keyPressedMap[button + MOUSE_OFFSET_CONSTANT] = true;
}

void KeyboardInput::mouseReleased(int x, int y, int button)
{
	// Update mousePressed Maps
	keyDownMap[button + MOUSE_OFFSET_CONSTANT] = false;
	keyReleasedMap[button + MOUSE_OFFSET_CONSTANT] = true;
}

void KeyboardInput::registerKeyCallback(KeyboardCallbacks* callbackInstance, CallbackType callbackType)
{
	set<KeyboardCallbacks*> * callbackSet;
	switch (callbackType) {
	case CALLBACK_PRESSED:
		callbackSet = &keyPressedCallbacks;
		break;
	case CALLBACK_RELEASED:
		callbackSet = &keyReleasedCallbacks;
		break;
	}

	if (callbackSet->count(callbackInstance) == 0) {
		callbackSet->insert(callbackInstance);
	}
}

void KeyboardInput::deregisterKeyCallback(KeyboardCallbacks* callbackInstance, CallbackType callbackType)
{
	set<KeyboardCallbacks*>* callbackSet;
	switch (callbackType) {
	case CALLBACK_PRESSED:
		callbackSet = &keyPressedCallbacks;
		break;
	case CALLBACK_RELEASED:
	default:
		callbackSet = &keyReleasedCallbacks;
		break;
	}

	if (callbackSet->count(callbackInstance) != 0) {
		callbackSet->erase(callbackInstance);
	}
}

void KeyboardInput::registerAlias(const string& alias, const string& str, int offset)
{
	// Edge case. Remap spacebar to underscore
	int key = convertStringToKey(str);
	keyAliasMappings[alias].push_back(key + offset);
}

bool KeyboardInput::queryInput(const string& alias, QueryType queryType)
{
	// Assume that Alias's that do not exist always return false
	if (keyAliasMappings.count(alias) == 0) return false;

	// Return true as soon as one of the keys for the binding have been triggered
	for (int& key : keyAliasMappings[alias]) {
		if (queryInput(key, queryType)) return true;
	}
	return false;
}

bool KeyboardInput::queryInput(const int key, QueryType queryType)
{
	/* If the key has not been pressed return false. Otherwise return
	what the value is. */
	switch (queryType) {
	case QUERY_PRESSED:
		return (keyPressedMap.count(key) == 0) ? false : keyPressedMap[key];
	case QUERY_DOWN:
		return (keyDownMap.count(key) == 0) ? false : keyDownMap[key];
	case QUERY_RELEASED:
		return (keyReleasedMap.count(key) == 0) ? false : keyReleasedMap[key];
	}

	cout << "KeyboardInput::queryInput: queryType not recognised. Returning false\n";
	return false;
}

void KeyboardInput::resetPollingMaps()
{
	keyPressedMap.clear();
	keyReleasedMap.clear();
}
