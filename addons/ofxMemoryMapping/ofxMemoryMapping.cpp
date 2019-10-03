#include "ofxMemoryMapping.h"

ofxMemoryMapping::ofxMemoryMapping() :
	fileSize(0), windowsFileHandle(NULL), windowsFileMapping(NULL), windowsFileView(NULL)
{
}

ofxMemoryMapping::ofxMemoryMapping(const string& filename)
{
	load(filename);
}

ofxMemoryMapping::~ofxMemoryMapping()
{
	if (windowsFileView) {
		printf("ofxMemoryMapping: Closing File '%s'.\n", filename.c_str());
		close();
	}
	//xinfo("~ofxMemoryMapping()");
}

void ofxMemoryMapping::close()
{
	/*
	https://docs.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-closehandle
	BOOL CloseHandle(
		HANDLE hObject
	);
	*/
	windowsCloseFileHandle();
	windowsCloseFileMappingHandle();
	windowsCloseFileView();
	fileSize = 0;
}

void ofxMemoryMapping::load(const string& filename, const size_t& fileSize)
{
	printf("ofxMemoryMapping: Loading file '%s'.\n", filename.c_str());

	windowsCreateFileHandle(filename);
	windowsCreateFileMappingHandle(fileSize);
	windowsCreateFileView();

	//info("load(string, size_t)");
}

void ofxMemoryMapping::windowsCloseFileHandle()
{
	if (CloseHandle(windowsFileHandle) == 0) {
		printf("ofxMemoryMapping: Error closing File.");
		info("windowsCloseFileHandle()");
	}
	windowsFileHandle = NULL;
}

void ofxMemoryMapping::windowsCloseFileMappingHandle()
{
	if (CloseHandle(windowsFileMapping) == 0) {
		printf("ofMemoryMapping: Error closing File Mapping.");
		info("windowsCloseFileMappingHandle()");
	}
	windowsFileMapping = NULL;
}

void ofxMemoryMapping::windowsCloseFileView()
{
	if (UnmapViewOfFile(windowsFileView) == 0) {
		printf("ofxMemoryMapping: Error closing File View.");
		info("windowsCloseFileView()");
	}
	windowsFileView = NULL;
}

void ofxMemoryMapping::windowsCreateFileHandle(const string& filename)
{
	this->filename = filename;
	/*
	This function opens the file for reading and writing. If the file does not exist it wil create a new one.
	It returns the file handle used for creating a file mapping.

	https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea?redirectedfrom=MSDN
	HANDLE CreateFileA(
		LPCSTR                lpFileName,
		DWORD                 dwDesiredAccess,
		DWORD                 dwShareMode,
		LPSECURITY_ATTRIBUTES lpSecurityAttributes,
		DWORD                 dwCreationDisposition,
		DWORD                 dwFlagsAndAttributes,
		HANDLE                hTemplateFile
	);
	*/
	windowsFileHandle = CreateFileA(
		filename.c_str(),
		(GENERIC_READ | GENERIC_WRITE),
		0,
		NULL,
		OPEN_ALWAYS, //CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL,
		NULL
	);

	/*
	This determines whether the existing file at this location was a larger size than what we specified and
	changes it accordinly. Eventually this should lead to abstracting away the need for specifying a file
	size when working view the the memory mapped files. Since the function returns the Lower order number
	and passes the higher order to the second parameter, some quick maths is required to combine the two.
	For our purposes this should not be neccessary but it's good to include.
	https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes

	https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfilesizeex
	https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-large_integer~r1
	BOOL GetFileSizeEx(
		HANDLE         hFile,
		PLARGE_INTEGER lpFileSize
	);
	*/
	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		printf("ofMemoryMapping: File '%s' already exists.\n", filename.c_str());

		LARGE_INTEGER actualFileSizeStruct;
		int result = GetFileSizeEx(windowsFileHandle, &actualFileSizeStruct);
		uint64_t actualFileSize = actualFileSizeStruct.QuadPart;

		if (!result) {
			printf("ofxMemoryMapping: GetFileSizeEx() Failed. Invalid File Size.\n");
			printf("\tLast Error: %d\n", GetLastError());
		}
		else if (actualFileSize > fileSize) {
			//printf("Existing file size of %llu is larger than specified %d. Increasing file allocation size.\n", actualFileSize, fileSize);
			fileSize = actualFileSize;
		}
	}
}

void ofxMemoryMapping::windowsCreateFileMappingHandle(const size_t& fileSize)
{
	this->fileSize = fileSize;
	/*
	This creates a file mapping. Here we have to specify the size of the file we are going to create. Past
	this point if we want to resize we need to recreate the entire mapping. Luckily this isn't a very
	expensive process but we try to reduce the amount of times we need to remap. We also need to convert
	the fileSize into it's lower and upper counterparts.

	https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createfilemappinga
	HANDLE CreateFileMappingA(
		HANDLE                hFile,
		LPSECURITY_ATTRIBUTES lpFileMappingAttributes,
		DWORD                 flProtect,
		DWORD                 dwMaximumSizeHigh,
		DWORD                 dwMaximumSizeLow,
		LPCSTR                lpName
	);
	*/

	uint64_t size = fileSize;
	uint32_t sizeLow = size & 0xffffffff;
	uint32_t sizeHigh = size >> 32;

	windowsFileMapping = CreateFileMappingA(
		windowsFileHandle,
		NULL,
		PAGE_READWRITE,
		sizeHigh,
		sizeLow,
		NULL // DON'T TRY AN BE SMART AND MAKE THIS SOMETHING YOU JUST WASTED 4 HOURS OF YOUR LIFE BECAUSE OF IT
	);

	if (!windowsFileMapping) {
		cout << "ofxMemoryMapping: Failed creating mapping HANDLE for " + filename << endl;
		cout << GetLastError() << endl;
		assert(false);
		return;
	}
}

void ofxMemoryMapping::windowsCreateFileView()
{
	/*
	This the the mmap function for Windows. We leave the parameters as 0 because we intend on using the entire file.

	https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile?redirectedfrom=MSDN
	LPVOID MapViewOfFile(
		HANDLE hFileMappingObject,
		DWORD  dwDesiredAccess,
		DWORD  dwFileOffsetHigh,
		DWORD  dwFileOffsetLow,
		SIZE_T dwNumberOfBytesToMap
	);
	*/

	windowsFileView = MapViewOfFile(
		windowsFileMapping,
		FILE_MAP_ALL_ACCESS,
		0,
		0,
		0
	);
	
	if (!windowsFileView) {
		cout << "Failed creating view pointer for " + filename << endl;
		cout << GetLastError() << endl;
		assert(false);
		return;
	}
}

void ofxMemoryMapping::load(const string& filename)
{
	load(filename, 1);
}

void ofxMemoryMapping::write(const void* data, const size_t& offset, const size_t& bytes)
{
	unsigned int newSize = offset + bytes;
	if (newSize > fileSize) {
		resize(newSize * 1.25);
	}
	memcpy((char*)windowsFileView + offset, data, bytes);
}

void ofxMemoryMapping::read(void* dataResult, const size_t& offset, const size_t& bytes) const
{
	memcpy(dataResult, (char*)windowsFileView + offset, bytes);
}

void ofxMemoryMapping::resize(const size_t& newSize)
{
	printf("ofxMemoryMapping: Resizing '%s' to: %d bytes.\n", filename.c_str(), newSize);

	windowsCloseFileMappingHandle();
	windowsCloseFileView();
	windowsCreateFileMappingHandle(newSize);
	windowsCreateFileView();

	info("resize(size_t)");
}

unsigned int ofxMemoryMapping::getFileSize() const
{
	return fileSize;
}

void ofxMemoryMapping::info(const string& func) const
{
	printf("ofxMemoryMapping::info() %s:\n", func.c_str());
	printf("\tLast Error: %d\n", GetLastError());
	printf("\tFile Handle for %s is %p\n", filename.c_str(), windowsFileHandle);
	printf("\tFile Mappin for %s is %p\n", filename.c_str(), windowsFileMapping);
	printf("\tFile View   for %s is %p\n", filename.c_str(), windowsFileView);
	printf("\tSize: %d\n", fileSize);
	SetLastError(0);
}
