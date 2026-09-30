using System.Runtime.CompilerServices;

namespace RadeSharp.Tests;

public class RadeCompTests
{
    [Fact]
    public void LayoutMatchesCRadeComp() => Assert.Equal(8, Unsafe.SizeOf<RadeComp>());
}
