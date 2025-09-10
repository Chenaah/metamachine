"""Cross-platform keyboard hit detection utility.

This module provides a KBHit class that allows non-blocking keyboard input
detection on both Windows and POSIX systems (Linux, macOS).

Original implementation distributed under GNU General Public License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""

import os

# Windows
if os.name == 'nt':
    import msvcrt

# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select


class KBHit:
    """Cross-platform keyboard hit detection class.
    
    Provides methods for non-blocking keyboard input detection and character reading.
    Automatically handles platform differences between Windows and POSIX systems.
    """

    def __init__(self):
        """Initialize KBHit object and set up platform-specific terminal settings."""

        if os.name == 'nt':
            pass

        else:

            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)


    def set_normal_term(self) -> None:
        """Reset terminal to normal mode.
        
        On Windows this is a no-op. On POSIX systems, restores original terminal settings.
        """

        if os.name == 'nt':
            pass

        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)


    def getch(self) -> str:
        """Get a single character from keyboard input.
        
        Should be called only after kbhit() returns True.
        Do not use in the same program as getarrow().
        
        Returns:
            Single character string
        """

        s = ''

        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')

        else:
            return sys.stdin.read(1)


    def getarrow(self) -> int:
        """Get arrow key code after kbhit() has been called.
        
        Should be called only after kbhit() returns True.
        Do not use in the same program as getch().
        
        Returns:
            Arrow key code:
            - 0: up arrow
            - 1: right arrow  
            - 2: down arrow
            - 3: left arrow
        """

        if os.name == 'nt':
            msvcrt.getch() # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]

        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]

        return vals.index(ord(c.decode('utf-8')))


    def kbhit(self) -> bool:
        """Check if a keyboard character is available to read.
        
        Returns:
            True if keyboard input is available, False otherwise
        """
        if os.name == 'nt':
            return msvcrt.kbhit()

        else:
            dr, dw, de = select([sys.stdin], [], [], 0)
            return dr != []


# Example usage
if __name__ == "__main__":
    """Example demonstrating KBHit usage."""
    kb = KBHit()
    
    print('Press any key to test keyboard input, or ESC to exit')
    print('This demonstrates non-blocking keyboard input detection.')
    
    try:
        while True:
            if kb.kbhit():
                c = kb.getch()
                if ord(c) == 27:  # ESC key
                    print('\nExiting...')
                    break
                print(f"Key pressed: '{c}' (ASCII: {ord(c)})")
    finally:
        kb.set_normal_term()  # Ensure terminal is reset
