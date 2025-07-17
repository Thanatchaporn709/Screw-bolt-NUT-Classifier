import * as React from "react";
import { AppBar, Toolbar, Typography, Button, Box } from "@mui/material";
import { useRouter } from "next/router";
import Link from "next/link";
import FunctionsIcon from "@mui/icons-material/Functions";
import useBearStore from "@/store/useBearStore";

const NavigationLayout = ({ children }) => {
  const router = useRouter();
  const appName = useBearStore((state) => state.appName);

  return (
    <>
      <AppBar position="sticky" sx={{ backgroundColor: "#D2F3EE" }}>
        <Toolbar sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          {/* Logo and App Name */}
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Link href={"/"}>
              <FunctionsIcon sx={{ color: "#7FB7BE", fontSize: "2rem", cursor: "pointer" }} />
            </Link>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 600,
                color: "#7FB7BE",
                marginLeft: 2,
                fontFamily: "Prompt",
              }}>
              {appName}
            </Typography>
          </Box>

          {/* Navigation Links */}
          <Box sx={{ display: "flex", gap: 3 }}>
            <NavigationLink href="/" label="Index" />
            <NavigationLink href="/page1" label="Main Page" />
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <main>{children}</main>
    </>
  );
};

const NavigationLink = ({ href, label }) => {
  return (
    <Link href={href} style={{ textDecoration: "none" }}>
      <Typography
        variant="body1"
        sx={{
          fontSize: "16px",
          fontWeight: 500,
          color: "#7FB7BE",
          padding: "8px 12px",
          borderRadius: 2,
          "&:hover": {
            backgroundColor: "#A9E4D8", // Lighter color for hover effect
            cursor: "pointer",
          },
        }}>
        {label}
      </Typography>
    </Link>
  );
};

export default NavigationLayout;
