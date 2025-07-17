import Head from "next/head";
import { Box, Typography, Button } from "@mui/material";
import { useRouter } from "next/router"; // Import the router for navigation

function Home() {
  const router = useRouter(); // Initialize the router

  // Function to navigate to "Page 1"
  const handleStartClick = () => {
    router.push("/page1"); // Navigate to the "Page 1"
  };

  return (
    <>
      <Head>
        <title>Home Page</title>
        <meta name="description" content="Index page with a start button" />
      </Head>

      <main>
        {/* Container Box */}
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
            height: "100vh",
            backgroundColor: "#FFE4E1",
            padding: 2,
          }}
        >
          <Typography
            variant="h3"
            sx={{
              fontWeight: "bold",
              color: "#F88379",
              marginBottom: 4,
              textAlign: "center",
            }}
          >
            LET'S GO TO SCREW/NUT DETECTION AREA
          </Typography>

          <Button
            variant="contained"
            color="primary"
            sx={{
              padding: "10px 20px",
              fontSize: "1.2rem",
              backgroundColor: "#FF69B4", // Pink color for the button
              "&:hover": {
                backgroundColor: "#FF1493", // Darker pink on hover
              },
            }}
            onClick={handleStartClick}
          >
            Start
          </Button>
        </Box>
      </main>
    </>
  );
}

export default Home;
