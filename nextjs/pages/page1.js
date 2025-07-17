import React from "react";
import { Box, Container, Paper, Typography } from "@mui/material";

export default function ScrewNutDetection() {
  return (
    <Container
      sx={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        height: "100vh",
        backgroundColor: "#FFE4E1",
        padding: 4,
        flexDirection: "column",
        backgroundImage: "url('https://www.example.com/screw_bolt_image.jpg')", // Replace this with the actual image URL
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      {/* Detection Area Header */}
      <Typography 
        variant="h1" 
        sx={{ 
          fontWeight: "bold", 
          color: "white", // Set text color to white
          marginBottom: 2,
          textAlign: "center"
        }}
      >
        Detection Area
      </Typography>
      
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          width: "100%",
        }}
      >
        {/* Detection Area */}
        <Paper
          sx={{
            width: "45%",
            height: "60vh",
            background: "linear-gradient(135deg, #FFB6C1 30%, #FFC0CB 90%)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            borderRadius: 3,
            boxShadow: 5,
            color: "white", // Text color inside Paper
          }}
        >
          <Typography variant="h6" color="white">Live Feed</Typography>
        </Paper>

        {/* Information Display */}
        <Box
          sx={{
            width: "45%",
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          <Paper
            sx={{
              padding: 2,
              background: "linear-gradient(135deg, #FFC0CB 30%, #FFB6C1 90%)",
              borderRadius: 3,
              boxShadow: 5,
            }}
          >
            <Typography variant="h5" sx={{ color: "#480607" }}>Type</Typography>
            <Typography variant="body1" sx={{ fontSize: "1.2rem", fontWeight: "bold", color: "white" }}>
              <span id="type">N/A</span>
            </Typography>
          </Paper>

          <Paper
            sx={{
              padding: 2,
              background: "linear-gradient(135deg, #FFC0CB 30%, #FFB6C1 90%)",
              borderRadius: 3,
              boxShadow: 5,
            }}
          >
            <Typography variant="h5" sx={{ color: "#480607" }}>Size</Typography>
            <Typography variant="body1" sx={{ fontSize: "1.2rem", fontWeight: "bold", color: "white" }}>
              <span id="size">N/A</span>
            </Typography>
          </Paper>

          <Paper
            sx={{
              padding: 2,
              background: "linear-gradient(135deg, #FFC0CB 30%, #FFB6C1 90%)",
              borderRadius: 3,
              boxShadow: 5,
            }}
          >
            <Typography variant="h5" sx={{ color: "#480607" }}>Length</Typography>
            <Typography variant="body1" sx={{ fontSize: "1.2rem", fontWeight: "bold", color: "white" }}>
              <span id="length">N/A</span>
            </Typography>
          </Paper>

          {/* History */}
          <Paper
            sx={{
              marginTop: 2,
              padding: 3,
              backgroundColor: "#fff",
              borderRadius: 2,
              boxShadow: 3,
            }}
          >
            <Typography variant="h5" sx={{ color: "#480607" }}>History</Typography>
            <ul id="history-list" style={{ fontSize: "1.1rem", paddingLeft: "20px", color: "white" }}>
              <li>No history yet</li>
            </ul>
          </Paper>
        </Box>
      </Box>
    </Container>
  );
}
